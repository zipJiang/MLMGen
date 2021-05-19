import argparse
from model.mlm import ConfigurationMLMSampler
from model.mlm import MODEL_NAME_DICT, SimpleMLMSampler


def parse_args():
    """This function will parse_args to get
    final argument for configurations.
    """
    parser = argparse.ArgumentParser(
        """Running the sampler in interactive model.
        """
    )
    parser.add_argument('--top_k', action='store', dest='top_k',
                        type=int, required=False, default=100)
    parser.add_argument('--min_length', action='store', dest='min_length',
                        type=int, required=False, default=20)
    parser.add_argument('--max_length', action='store', dest='max_length',
                        type=int, required=False, default=40)
    parser.add_argument('--batch_size', action='store', dest='batch_size',
                        type=int, required=False, default=32)
    parser.add_argument('--device', action='store', dest='device',
                        type=str, required=False, default='cpu')
    parser.add_argument('--top_p', action='store', dest='top_p',
                        type=float, required=False, default=.92)
    parser.add_argument('--temperature', action='store', dest='temperature',
                        type=float, required=False, default=.6)
    parser.add_argument('--sampling_round', action='store', dest='sampling_round',
                        type=int, required=False, default=500)
    parser.add_argument('--burn_in_rounds', action='store', dest='burn_in_rounds',
                        type=int, required=False, default=100)
    parser.add_argument('--model_name', action='store', dest='model_name',
                        type=str, required=False, default='bert',
                        choices=MODEL_NAME_DICT.keys())
    parser.add_argument('--cache_dir', action='store', dest='cache_dir',
                        type=str, required=False, default='cache/')
    
    args = parser.parse_args()
    
    return args


def main():
    """Main function for doing interactive scoring.
    """
    args = parse_args()
    
    configuration = ConfigurationMLMSampler(length_range=(args.min_length, args.max_length),
                                            sampling_rounds=args.sampling_round,
                                            burn_in_rounds=args.burn_in_rounds,
                                            temperature=args.temperature,
                                            top_k=args.top_k,
                                            top_p=args.top_p,
                                            batch_size=args.batch_size,
                                            device=args.device,
                                            ckpt_name=MODEL_NAME_DICT[args.model_name],
                                            cache_dir=args.cache_dir)
    
    # construct a model
    model = SimpleMLMSampler(configuration)
    
    # now running sampling using the model.
    while True:
        word = input("Input a word / phrase to generate lexically constrained sentences([QUIT]) to quit:")
        for candidate in model.sample_sentences(num_samples=1, word=word):
            print(candidate)

        
if __name__ == '__main__':
    main()