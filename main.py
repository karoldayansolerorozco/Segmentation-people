import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb  # 👈 Nuevo import

from UNet import UNet
from dataloader import PeopleDataset

if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 2
    EPOCHS = 2
    DATA_PATH = "people_data"
    MODEL_SAVE_PATH = "unet.pth"
    MAX_SAMPLES = 100

    # Inicializar W&B
    wandb.init(
        project="people-segmentation",
        name="unet-baseline",
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "model": "UNet"
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 Usando dispositivo: {device}")

    # Dataset
    print("📦 Cargando dataset completo...")
    full_dataset = PeopleDataset(root_path=DATA_PATH, split="train")
    full_dataset = full_dataset
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [0.8, 0.2], generator=generator)
    print(f"✔ Dataset reducido: {len(train_dataset)} para entrenamiento, {len(val_dataset)} para validación.")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Modelo
    print("🧠 Inicializando modelo UNet...")
    model = UNet(in_channels=3, num_classes=1).to(device)

    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print("✔ Pesos cargados correctamente desde unet.pth")
    except Exception as e:
        print(f"⚠ No se cargaron pesos preentrenados: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Entrenamiento
    print("🚀 Comenzando entrenamiento...")
    for epoch in range(EPOCHS):
        print(f"\n🔁 Época {epoch + 1}/{EPOCHS}")
        model.train()
        train_running_loss = 0.0

        for idx, (img, mask) in enumerate(train_dataloader):
            if idx % 2 == 0:
                print(f"   ⏳ Batch {idx + 1}/{len(train_dataloader)}")

            img = img.float().to(device)
            mask = mask.float().to(device)

            y_pred = model(img)
            loss = criterion(y_pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

        train_loss = train_running_loss / (idx + 1)
        print(f"✅ Entrenamiento completado para época {epoch + 1}. Loss: {train_loss:.4f}")

        # Validación con IoU
        print(f"🔍 Validando modelo...")
        model.eval()
        val_running_loss = 0.0
        iou_total = 0.0
        iou_batches = 0

        with torch.no_grad():
            for idx, (img, mask) in enumerate(val_dataloader):
                img = img.float().to(device)
                mask = mask.float().to(device)

                y_pred = model(img)
                loss = criterion(y_pred, mask)
                val_running_loss += loss.item()

                preds = torch.sigmoid(y_pred) > 0.5
                preds = preds.bool()
                mask = mask.bool()

                intersection = (preds & mask).float().sum((1, 2, 3))
                union = (preds | mask).float().sum((1, 2, 3))
                iou = (intersection / (union + 1e-6)).mean().item()
                iou_total += iou
                iou_batches += 1

                # Subir ejemplos visuales a W&B (solo del primer batch)
                if epoch == 0 and idx == 0:
                    wandb.log({
                        "Ejemplo entrada": wandb.Image(img[0].cpu()),
                        "Máscara verdadera": wandb.Image(mask[0].cpu()),
                        "Predicción": wandb.Image(preds[0].cpu())
                    })

        val_loss = val_running_loss / (idx + 1)
        mean_iou = iou_total / iou_batches

        # Registrar métricas en W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": mean_iou
        })

        print(f"📊 Validación completada. Loss: {val_loss:.4f}, IoU: {mean_iou:.4f}")
        print("-" * 30)

    # Guardar pesos del modelo entrenado
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Modelo guardado exitosamente en {MODEL_SAVE_PATH}")

    # Finalizar experimento W&B
    wandb.finish()
