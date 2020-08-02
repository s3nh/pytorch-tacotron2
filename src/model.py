import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence 

class Encoder(nn.Module):

    def __init__(self, num_chars, padding_idx, embedding_dim=512, encoder_num_convs=3, 
            kernel_size=5, hidden_size=256, bidirectional=True):

        super(Encoder, self).__init__()
        #Embedding layer with prededfined embedding params 
        
        self.embedding = nn.Embedding(num_chars, embedding_dim, padding_idx=padding_idx)
        padding = (kernel -1) // 2 
        convs = []
        
        for _ in range(encoder_num_convs):
            convs += [ConvBlock(embedding_dim, embedding_dim, kernel_size, padding, 'relu')]
        
        # Repack 
        self.convs = nn.Sequential(*convs)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers =1, batch_first = True, bidirectional =bidirectional)
        
    
    def forward(self, text_padded, input_lengths):
        """
        Args:
            text_padded:
                N -> batch_size
                T -> sequence length
        """
        x = self.embedding(text_padded) # [N, T, D]
        x = x.transpose(1, 2)
        x = self.convs(x)
        x = x.transpose(1, 2)

        total_length = x.size(1) 
        packed_input = pack_padded_sequence(x, input_lengths, batch_first=True)

        self.rnn.flatten_parameters()
        packed_output, _ = self.rnn(packed_input)
        ouput, _ = pad_packed_sequence(packed_output, batch_first = True, total_length = total_length)
        output

class Decoder(nn.Module):
    # Mel spectrogram prediction 

    def __init__(self, feature_dim, encoder_hidden_size=512, 
            prenet_dim=256, decoder_hidden_size = 1024, attention_dim=128, 
            location_feature_dim=32, postnet_num_convs=5, postnet_filter_size=512, postnet_kernel_size=5, max_decoder_step=1000):
        super(Decoder, self).__init__()

        self.feature_dim = feature_dim
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.max_decoder_steps = max_decoder_steps
            

        self.prenet = PreNet(feature_dim, prenet_dim)
        self.rnn = nn.ModuleList()
        self.rnn += [nn.LSTMCell(prenet_dim + encoder_hidden_size, decoder_hidden_size)]
        self.rnn += [nn.LSTMCell(decoder_hidden_size, decoder_hidden_size)]

        self.attention = LocationSensitiveAttention(attention_dim, 
                decoder_hidden_size, encoder_hidden_size, location_feature_dim)
        
        self.feature_linear = nn.Linear(decoder_hidden_size + encoder_hidden_size, feature_dim)
        self.stop_linear = nn.Linear(decoder_hidden_size + encoder_hidden_size, 1)
        
        self.postnet = PostNet(feature_dim, postnet_num_convs, postnet_filter_size, postnet_kernel_size)
        

        def forward(self, encoder_padded_outputs, encoder_mask, feat_padded, decoder_mask):
            # Create empty tensor of encoder padded output shape 
            go_frame = self._init_go_frame(encoder_padded_outputs)
            expand_feat = torch.cat((go_frame, feat_padded), dim=1) #[N, To+1, D]

            # Rnn state initialization
            self._init_state(encoder_padded_outputs)
            self.encoder_mask = encoder_mask 

            # Forward part 

            feat_outputs, stop_tokens, attention_weights = [], [], [] 
            To = feat_padded.size(1)
            
            prenet_out = self.prenet(expand_feat)
            
            for t in range(To):
                if t == 0:
                    self.attention.reset()
                step_input = prenet_out[:, t, :]
                feat_output, stop_token, attention_weight = self._step(step_input)
                feat_outputs += [feat_output]
                stop_okens += [stop_token]
                attention_weights += [attention_weight]
                
            feat_outputs = torch.stack(feat_outputs, dim=1)
            stop_tokens = torch.stack(stop_tokens, dim=1).squeeze()
            attention_weights = torch.stack(attention_weights, dim=1)
            feat_residual_outputs = self.postnet(feat_outputs)


            # Mask part

            decoder_mask = decoder_mask.unsqueeze(-1) 
            feat_outputs = feat_outputs.masked_fill(decoder_mask, 0.0)
            stop_tokens = stop_tokens.masked_fill(decoder_mask.squeeze(), 1e3)
            return feat_outputs, feat_residual_outputs, stop_tokens, attention_weights


        


class PreNet(nn.Module):
    """The prediction from the previous time step is first passed through a small pre-net containing 2 fully connected layers of 256 hidden ReLU units. We found that the pre-net acting as an information bottleneck was essential for learning attention."""
    def __init__(self, feature_dim, prenet_dim=256, p=0.5):
        super(PreNet, self).__init__()
        self.linear1 = nn.Linear(feature_dim, prenet_dim)
        self.linear2 = nn.Linear(prenet_dim, prenet_dim)
        self.p = p # dropout_rate

    def forward(self, x):
        """
        Args:
            x: [N, T, D], D is input dim / [N, D]
        Returns:
            [N, T, H], H is hidden unit / [N, H]
        """
        x = F.dropout(F.relu(self.linear1(x)), p=self.p, training=True)
        x = F.dropout(F.relu(self.linear2(x)), p=self.p, training=True)
        return x


class LocationSensitiveAttention(nn.Module):
    def __init__(self, attention_dim=128, decoder_hidden_size=1024,
                 encoder_hidden_size=512, location_feature_dim=32):
        super(LocationSensitiveAttention, self).__init__()
        self.W = nn.Linear(decoder_hidden_size, attention_dim, bias=True) # keep one bias
        self.V = nn.Linear(encoder_hidden_size, attention_dim, bias=False)
        self.U = nn.Linear(location_feature_dim, attention_dim, bias=False)
        self.F = nn.Conv1d(in_channels=1, out_channels=location_feature_dim,
                           kernel_size=31, stride=1, padding=(31-1)//2,
                           bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.reset()

    def reset(self):
        """Remember to reset at decoder step 0"""
        self.Vh = None # pre-compute V*h_j due to it is independent from the decoding step i

    def _cal_energy(self, query, values, cumulative_attention_weights, mask=None):
        """Calculate energy:
           e_ij = score(s_i, ca_i-1, h_j) = v tanh(W s_i + V h_j + U f_ij + b)
           where f_i = F * ca_i-1,
                 ca_i-1 = sum_{j=1}^{T-1} a_i-1
        Args:
            query: [N, Hd], decoder state
            values: [N, Ti, He], encoder hidden representation
        Returns:
            energies: [N, Ti]
        """
        # print('query', query.size())
        # print('values', values.size())
        query = query.unsqueeze(1) #[N, 1, Hd], insert time-axis for broadcasting
        Ws = self.W(query) #[N, 1, A]
        if self.Vh is None:
            self.Vh = self.V(values) #[N, Ti, A]
        location_feature = self.F(cumulative_attention_weights.unsqueeze(1)) #[N, 32, Ti]
        # print(location_feature.size())
        Uf = self.U(location_feature.transpose(1, 2)) #[N, Ti, A]
        energies = self.v(torch.tanh(Ws + self.Vh + Uf)).squeeze(-1) #[N, Ti]
        # print('W s_i', Ws.size())
        # print('V h_j', self.Vh.size())
        # print('U f_ij', Uf.size())
        # print('mask', mask)
        # print('energies', energies)
        if mask is not None:
            energies = energies.masked_fill(mask, -np.inf)
        # print(energies)
        return energies

    def forward(self, query, values, cumulative_attention_weights, mask=None):
        """
        Args:
            query: [N, Hd], decoder state
            values: [N, Ti, He], encoder hidden representation
            mask: [N, Ti]
        Returns:
            attention_context: [N, He]
            attention_weights: [N, Ti]
        """
        energies = self._cal_energy(query, values, cumulative_attention_weights, mask) #[N, Ti]
        attention_weights = F.softmax(energies, dim=1) #[N, Ti]
        # print('weights', attention_weights)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), values) #[N, 1, Ti] bmm [N, Ti, He] -> [N, 1, He]
        attention_context = attention_context.squeeze(1) # [N, Ti]
        # print('context', attention_context.size())
        return attention_context, attention_weights


class PostNet(nn.Module):
    """Finally, the predicted mel spectrogram is passed through a 5-layer convolutional post-net which predicts a residual to add to the prediction to improve the overall reconstruction. Each post-net layer is comprised of 512 filters with shape 5 × 1 with batch normalization, followed by tanh activations on all but the final layer."""
    def __init__(self, feature_dim, postnet_num_convs=5, postnet_filter_size=512, postnet_kernel_size=5):
        super(PostNet, self).__init__()
        padding = (postnet_kernel_size - 1) // 2  # keep length unchanged
        convs = [ConvBlock(feature_dim, postnet_filter_size, postnet_kernel_size, padding, 'tanh')]
        for _ in range(postnet_num_convs-2):
            convs += [ConvBlock(postnet_filter_size, postnet_filter_size, postnet_kernel_size, padding, 'tanh')]
        convs += [ConvBlock(postnet_filter_size, feature_dim, postnet_kernel_size, padding, None)]
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """
        Args:
            x: [N, T, D]
        Returns:
            out = [N, T, D]
        """
        x = x.transpose(1, 2)
        out = self.convs(x)
        out = out.transpose(1, 2)
        return out


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm1d -> (nonlinear) -> Dropout"""
    def __init__(self, in_channels, out_channels, kernel_size, padding, nonlinear=None):
        super(ConvBlock, self).__init__()
        conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        norm = nn.BatchNorm1d(out_channels)
        # "The convolutional layers in the network are regularized using dropout with probability 0.5"
        dropout = nn.Dropout(p=0.5)
        if nonlinear == 'relu':
            relu = nn.ReLU()
            self.net = nn.Sequential(conv1d, norm, relu, dropout)
        elif nonlinear == 'tanh':
            tanh = nn.Tanh()
            self.net = nn.Sequential(conv1d, norm, tanh, dropout)
        else:
            self.net = nn.Sequential(conv1d, norm, dropout)
    
    def forward(self, x):
        """
        Args:
            x: [N, Ci, T], N is batch size, C is channle size, T is sequence length
        Returns:
            output: [N, Co, T]
        """
        output = self.net(x)
        return output


if __name__ == "__main__":
    torch.manual_seed(223)
    N, Ti, To = 3, 6, 8
    num_chars, padding_idx, feature_dim = 10, 0, 5
    text_padded = torch.randint(num_chars, (N, Ti)).long()
    text_padded[-2, -2:] = padding_idx
    text_padded[-1, -4:] = padding_idx
    input_lengths = torch.LongTensor([Ti, Ti-2, Ti-4])
    feat_padded = torch.randint(5, (N, To, feature_dim))
    feat_padded[-2, -1:] = 0
    feat_padded[-1, -3:] = 0
    feat_lengths = torch.LongTensor([To, To-1, To-3])
    print(text_padded)
    print(input_lengths)
    print(feat_padded)
    print(feat_lengths)

    fpn = FeaturePredictNet(num_chars, padding_idx, feature_dim)
    feat_outputs, feat_residual_outputs, stop_tokens, attention_weights \
        = fpn(text_padded, input_lengths, feat_padded, feat_lengths)
    print(fpn)
    print('feat_outputs', feat_outputs.size())
    print('feat_outputs', feat_outputs)
    print('stop_tokens', stop_tokens.size())
    print('stop_tokens', stop_tokens)
    print('feat_residual_outputs', feat_residual_outputs.size())
    print('feat_residual_outputs', feat_residual_outputs)
    print('attention weights', attention_weights.size())
    print('attention weights', attention_weights)

    encoder = Encoder(num_chars, padding_idx)
    encoder_padded_outputs = encoder(text_padded, input_lengths)
    encoder_padded_pred = encoder_padded_outputs.clone()
    print('encoder_padded_outputs', encoder_padded_outputs)
    # loss = nn.MSELoss()(encoder_padded_outputs, encoder_padded_pred)
    # loss.backward()
    # print(loss)

    encoder_padded_outputs2 = encoder_padded_outputs.clone()
    decoder = Decoder(feature_dim)
    feat_outputs, feat_residual_outputs, stop_tokens, attention_weights \
        = decoder(encoder_padded_outputs2, input_lengths, feat_padded, feat_lengths)
    loss = nn.MSELoss()(feat_outputs, feat_padded)
    loss.backward()
    print(loss)

