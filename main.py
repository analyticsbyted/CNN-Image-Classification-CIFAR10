import argparse
from train import train_model
from evaluate import evaluate_model
from prediction import predict_image
import os

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Image Classification')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'predict'],
                        help='Mode to run the script in. Can be train, evaluate, or predict.')
    parser.add_argument('--image_path', type=str, help='Path to the image to predict.')

    args = parser.parse_args()

    if args.mode == 'train':
        print("Training model...")
        train_model()
        print("Training complete.")
    elif args.mode == 'evaluate':
        print("Evaluating model...")
        evaluate_model()
        print("Evaluation complete.")
    elif args.mode == 'predict':
        if args.image_path:
            predicted_class, confidence_score = predict_image(args.image_path)
            class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            print(f"Image: {args.image_path}, Predicted class: {class_labels[predicted_class]}, Confidence: {confidence_score:.2f}")
        else:
            print("Please provide an image path using the --image_path argument for prediction.")

if __name__ == '__main__':
    main()
