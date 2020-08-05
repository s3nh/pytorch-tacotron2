import argparse 
import os 
import librosa
import matplotlib.pyplot as plt 
import torch

from utils.audio_process import inv_spectrogram, save_wav
from src.model import FeaturePredictNet
from utils.text_process import text_to_sequence


def create_args():
    # What do wee need 
    # model path text path out file 
    # Cuda or not, change for times 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str)
    parser.add_argument('--text_file', type = str)
    parser.add_argument('--out_dir', type = str)
    parser.add_argument('--use_cuda', type = int)
    args = parser.parse_args()
    return args

def synthesis(args):

    model = FeaturePredictNet.load_model(args.model_path)
    model.eval()
    model.cuda()

    os.makedirs(args.out_dir, exist_ok=True)

    # Did not use grad 
    with torch.no_grad():

        with open(args.text_file, 'r') as text_file:
            for i, text in enumerate(text_file.readlines()):
                filename = str(i)
                print(filename)
                print(text)
                text = torch.LongTensor(text_to_sequence(text)).unsqueeze(0)
                input_length = torch.LongTensor([text.size(-1)])
                text, input_length = text.cuda(), input_length.cuda()
                outputs = model.inference(text, input_length)
                feat_outputs, feat_residual_outputs, _, attention_weights = outputs
                feat_pred = feat_outputs + feat_residual_outputs
                
                audio = inv_spectrogram(feat_pred[0].cpu().numpy().T)
                audio_path = os.path.join(args.out_dir, f'{filename}.wav')
                print(audio_path)
                save_wav(audio, audio_path)

def main():
    args =  create_args()
    synthesis(args)



if __name__ == "__main__":
    main()
