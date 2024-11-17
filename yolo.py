import argparse
from ultralytics import YOLO

# Define classes for your dataset
CLASSES = [
    "UNKNOWN", "VEHICLE", "LARGE_VEHICLE", "BICYCLE", "PEDESTRIAN",
    "BARRICADE", "ANIMAL", "RAILROAD", "TRAFFIC_SIGN", "TRAFFIC_LIGHT"
]

def start_train(model_size, dataset_path, epochs, img_size):
    """
    Trains a YOLO model on the provided dataset.
    
    Parameters:
    - model_size (str): Model size, e.g., 'nano', 'small', 'medium'.
    - dataset_path (str): Path to the KITTY-formatted dataset.
    - epochs (int): Number of training epochs.
    - img_size (int): Image size for training.
    - resume (bool): If True, resume training from the last checkpoint.
    """
    
    model_name = f'yolov10{model_size}.pt'  # Update to 'yolov11' if needed
    model = YOLO(model_name)
    print(1)
    model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=img_size,
        device='cpu',
        project='out/checkpoint1',
    )

def continue_train(model_size, dataset_path, epochs, img_size, con=1):
    model_name = f'out/checkpoint{con}/train/weights/best.pt'
    # Load or initialize model
    # size = {md:s for md in ["nano", "small", "medium"]}
    model = YOLO(model_name)

    model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=img_size,
        device='cpu',
        project= f'out/checkpoint{con+1}',
            #classes=len(CLASSES),
            #names=CLASSES,
            # project='out',
            # name='yolov10'
    )

def ov_convert(cp):
    model_name = f'out/checkpoint{cp}/train/weights/best.pt'
    model = YOLO(model_name)
    model.export(format="openvino")

    ov = f'out/checkpoint{cp}/train/weights/best_openvino_model/'
    ov_model = YOLO(ov)

    ov_model("https://ultralytics.com/images/bus.jpg", save = True, show = True, project= f'out/checkpoint{cp}')
    


def main():
    parser = argparse.ArgumentParser(description="Train a YOLO model on a custom dataset.")
    parser.add_argument("--model_size", type=str, choices=["n", "s", "m"],
                        help="Size of the YOLO model to train: n, s, or m.")
    parser.add_argument("--dataset", type=str, 
                        help="Path to the dataset in KITTY format.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50).")
    parser.add_argument("--img_size", type=int, default=640,
                        help="Image size for training (default: 640).")
    parser.add_argument("--con_train", type=int, default=0,
                        help="Continue training from the last checkpoint if available.")
    parser.add_argument("--ov_convert", action="store_true",
                        help="Convert from given checkpoint if available.")

    args = parser.parse_args()

    # Start training
    if args.ov_convert and args.con_train > 0:
        ov_convert(args.con_train)
    elif args.con_train > 0:
        continue_train(args.model_size, args.dataset, args.epochs, args.img_size, args.con_train)
    else:
        start_train(args.model_size, args.dataset, args.epochs, args.img_size)

if __name__ == "__main__":
    main()
