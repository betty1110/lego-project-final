import torch
from .classifier_cnn import LegoCNN
from .mapping import pieceMapping, colorMapping
from .transformations import get_test_transforms

# Predicts the piece ID and color ID of a LEGO image
def predict(img):
    model = LegoCNN(num_pieceid=200, num_colorid=40)  # initialize model

    checkpoint = torch.load('files/weights/classifier.pth')  # load weights
    model.load_state_dict(checkpoint['model_state_dict'])  # load state dict
    model.eval()  # set model to eval mode

    resizer = get_test_transforms()  # get preprocessing transforms
    input_img = resizer(img).unsqueeze(0)  # preprocess and add batch dim

    # Run inference without gradient computation
    with torch.no_grad():
        dict = model(input_img)  # get predictions

    out_piece = dict['pieceid']  # piece ID logits
    out_color = dict['colorid']  # color ID logits

    # Get predicted class indices
    pred_piece = torch.argmax(out_piece, dim=1).item()
    pred_color = torch.argmax(out_color, dim=1).item()

    # Map indices back to original labels
    invPieceMapping = {v: k for k, v in pieceMapping.items()}
    invColorMapping = {v: k for k, v in colorMapping.items()}

    return invPieceMapping[pred_piece], invColorMapping[pred_color]
