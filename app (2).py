
import torch
import torch.nn as nn
import json
import numpy as np
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import os

# ── Model Architecture ──────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, feature_dim=2048, hidden_size=512):
        super().__init__()
        self.fc      = nn.Linear(feature_dim, hidden_size)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        return self.dropout(self.relu(self.fc(x)))

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512, num_layers=2, pad_idx=0):
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm        = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)
        self.fc          = nn.Linear(hidden_size, vocab_size)
        self.dropout     = nn.Dropout(0.5)
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
    def forward(self, captions, enc_out):
        embeds = self.dropout(self.embedding(captions))
        h0 = enc_out.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(embeds, (h0, c0))
        return self.fc(out)

class Seq2SeqCaptioner(nn.Module):
    def __init__(self, vocab_size, feature_dim=2048, embed_dim=256, hidden_size=512, num_layers=2):
        super().__init__()
        self.encoder = Encoder(feature_dim, hidden_size)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_size, num_layers)
    def forward(self, features, captions):
        return self.decoder(captions[:, :-1], self.encoder(features))

# ── Load vocab & config ──────────────────────────────────────────────
with open("vocab.json") as f:
    data = json.load(f)
word2idx = data["word2idx"]
idx2word = {int(k): v for k, v in data["idx2word"].items()}

with open("config.json") as f:
    cfg = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Seq2SeqCaptioner(
    vocab_size  = cfg["vocab_size"],
    embed_dim   = cfg["embed_dim"],
    hidden_size = cfg["hidden_size"],
    num_layers  = cfg["num_layers"]
).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ── ResNet50 feature extractor ──────────────────────────────────────
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

START_IDX = word2idx["<start>"]
END_IDX   = word2idx["<end>"]
PAD_IDX   = word2idx["<pad>"]
NUM_LAYERS = cfg["num_layers"]

def extract_features(pil_img):
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(tensor).view(1, -1)
    return feat

def greedy(feature, max_len=50):
    enc_out = model.encoder(feature)
    h = enc_out.unsqueeze(0).repeat(NUM_LAYERS,1,1)
    c = torch.zeros_like(h)
    word_id = torch.tensor([[START_IDX]], device=device)
    result = []
    for _ in range(max_len):
        embed = model.decoder.dropout(model.decoder.embedding(word_id))
        out, (h,c) = model.decoder.lstm(embed, (h,c))
        word_id = model.decoder.fc(out.squeeze(1)).argmax(-1, keepdim=True)
        word = idx2word[word_id.item()]
        if word == "<end>": break
        if word not in ["<start>","<pad>","<unk>"]:
            result.append(word)
    return " ".join(result)

def beam(feature, beam_width=5, max_len=50):
    enc_out = model.encoder(feature)
    h0 = enc_out.unsqueeze(0).repeat(NUM_LAYERS,1,1)
    c0 = torch.zeros_like(h0)
    beams = [(0.0, [START_IDX], h0, c0)]
    completed = []
    for _ in range(max_len):
        new_beams = []
        for score, tokens, h, c in beams:
            last = torch.tensor([[tokens[-1]]], device=device)
            embed = model.decoder.dropout(model.decoder.embedding(last))
            out, (hn, cn) = model.decoder.lstm(embed, (h, c))
            lp = torch.log_softmax(model.decoder.fc(out.squeeze(1)), -1)
            topk_s, topk_i = lp.topk(beam_width)
            for s, idx in zip(topk_s[0], topk_i[0]):
                nt = tokens + [idx.item()]
                if idx.item() == END_IDX:
                    completed.append((score + s.item()) / len(nt), nt)
                else:
                    new_beams.append((score + s.item(), nt, hn, cn))
        if not new_beams: break
        new_beams.sort(key=lambda x: x[0]/len(x[1]), reverse=True)
        beams = new_beams[:beam_width]
    if not completed:
        completed = [(s/len(t), t) for s,t,_,_ in beams]
    best = max(completed, key=lambda x: x[0])[1]
    return " ".join(idx2word[i] for i in best if i not in [START_IDX, END_IDX, PAD_IDX])

def caption_image(image, method, beam_width):
    if image is None:
        return "Please upload an image.", ""
    feat = extract_features(image)
    if method == "Greedy Search":
        cap = greedy(feat)
    else:
        cap = beam(feat, beam_width=int(beam_width))
    return cap

with gr.Blocks(title="Neural Storyteller") as demo:
    gr.Markdown("## Neural Storyteller — Image Captioning")
    gr.Markdown("Upload any image and the model will generate a natural language description.")
    with gr.Row():
        with gr.Column():
            img_input  = gr.Image(type="pil", label="Upload Image")
            method     = gr.Radio(["Greedy Search", "Beam Search"], value="Beam Search", label="Decoding method")
            beam_width = gr.Slider(2, 10, value=5, step=1, label="Beam width (for Beam Search)")
            btn        = gr.Button("Generate Caption", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="Generated Caption", lines=3)

    btn.click(fn=caption_image, inputs=[img_input, method, beam_width], outputs=output)

    gr.Examples(
        examples=[[f"sample_images/{f}"] for f in os.listdir("sample_images")][:5]
            if os.path.exists("sample_images") else [],
        inputs=img_input
    )

demo.launch()
