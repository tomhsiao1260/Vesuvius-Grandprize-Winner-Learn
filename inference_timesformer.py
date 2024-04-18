from tap import Tap

class InferenceArgumentParser(Tap):
    model_path:str= 'checkpoints/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt'
    
args = InferenceArgumentParser().parse_args()

if __name__ == "__main__":
    print(args.model_path)