import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import your project modules
from src.utils.config import get_default_configs
from src.data.parser import load_corpus
from src.data.split import split_train_val_test
from src.data.dataset import build_datasets, TextAgeDataset
from src.models.cnn import AgeCNN
from src.data.features import FormFeatureExtractor

def main():
    print("Loading configurations...")
    data_cfg, feat_cfg, model_cfg, train_cfg = get_default_configs()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 SUCCESS: NVIDIA GPU detected! Training on: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ WARNING: No GPU detected by PyTorch. Falling back to CPU.")

    print("\nLoading the full corpus...")
    entries = load_corpus(data_cfg.corpus_dir)
    
    # 1. 85/15/10 Split
    train_entries, val_entries, test_entries = split_train_val_test(
        entries, 
        test_size=0.10, 
        val_size=0.15, 
        random_seed=data_cfg.random_seed
    )
    
    print(f"\n📊 Data Distribution:")
    print(f"   - Train : {len(train_entries)} texts")
    print(f"   - Val   : {len(val_entries)} texts")
    print(f"   - Test  : {len(test_entries)} texts")

  
   # 2. Build datasets
    print("\nBuilding Datasets (Feature extraction in progress)...")
    
    # We just pass the entire feat_cfg object directly! Clean and simple.
    my_extractor = FormFeatureExtractor(config=feat_cfg)
    
    train_ds, val_ds = build_datasets(
        train_entries, 
        val_entries,
        extractor=my_extractor,
        sequence_length=data_cfg.sequence_length,
        train_stride=data_cfg.stride
    )
    
    test_ds = TextAgeDataset(
        test_entries, 
        extractor=my_extractor, 
        sequence_length=data_cfg.sequence_length, 
        stride=data_cfg.sequence_length
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False)

    # 4. Initialize Model, Loss, Optimizer
    model = AgeCNN(model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # The optimizer now perfectly reads your updated weight_decay from config.py!
    optimizer = optim.Adam(
        model.parameters(), 
        lr=train_cfg.learning_rate, 
        weight_decay=train_cfg.weight_decay  
    )
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=train_cfg.lr_scheduler_factor, patience=train_cfg.lr_scheduler_patience)

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(train_cfg.checkpoint_dir, "tensorboard_logs")
    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # 5. Training Loop
    print("\nStarting training loop...")
    for epoch in range(train_cfg.num_epochs):
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss, correct_train, train_mae_sum, total_train = 0.0, 0, 0.0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.num_epochs} [Train]", leave=False)
        
        for features, labels in train_pbar:
            features, labels = features.to(device), labels.to(device)
            # labels currently represent 0-40. We divide by 10 to get the decade.
            # torch.clamp ensures age 70 (which becomes 4) gets pushed into bracket 3.
            labels = labels // 10
            labels = torch.clamp(labels, 0, 3)
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            
            # ANTI-OVERFITTING FIX: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(logits, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_mae_sum += torch.abs(predicted - labels).sum().item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / total_train
        train_acc = 100 * correct_train / total_train
        train_mae = train_mae_sum / total_train
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, correct_val, val_mae_sum, total_val = 0.0, 0, 0.0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{train_cfg.num_epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for features, labels in val_pbar:
                features, labels = features.to(device), labels.to(device)

                labels = labels // 10
                labels = torch.clamp(labels, 0, 3)



                logits = model(features)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(logits, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_mae_sum += torch.abs(predicted - labels).sum().item()
                
        avg_val_loss = val_loss / total_val
        val_acc = 100 * correct_val / total_val
        val_mae = val_mae_sum / total_val
        
        # TensorBoard Logs
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('MAE/Train', train_mae, epoch)
        writer.add_scalar('MAE/Validation', val_mae, epoch)

        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{train_cfg.num_epochs}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train MAE: {train_mae:.2f} yrs")
        print(f"             | Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Val MAE:   {val_mae:.2f} yrs")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(train_cfg.checkpoint_dir, "best_model.pth"))
            print("  -> ✨ New best model saved!")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg.patience:
                print(f"\n🛑 Early Stopping triggered at epoch {epoch+1}")
                break

    writer.close()

   # 6. FINAL TEST SET EVALUATION
    print("\n" + "="*50)
    print("🎓 FINAL EVALUATION ON UNSEEN TEST DATA (10%)")
    print("="*50)
    
    best_model_path = os.path.join(train_cfg.checkpoint_dir, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    
    test_loss, correct_test, test_mae_sum, total_test = 0.0, 0, 0.0, 0
    
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Final Test"):
            features, labels = features.to(device), labels.to(device)

            labels = labels // 10
            labels = torch.clamp(labels, 0, 3)

            logits = model(features)
            loss = criterion(logits, labels)
            
            test_loss += loss.item() * features.size(0)
            _, predicted = torch.max(logits, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            test_mae_sum += torch.abs(predicted - labels).sum().item()
            
            # ON SAUVEGARDE LES PRÉDICTIONS DE CE BATCH
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_test_loss = test_loss / total_test
    test_acc = 100 * correct_test / total_test
    test_mae = test_mae_sum / total_test
    
    print(f"\nFinal Results (Chunk Level):")
    print(f"🎯 Accuracy : {test_acc:.2f}%")
    print(f"📏 MAE      : {test_mae:.2f} decades")
    print(f"📉 Loss     : {avg_test_loss:.4f}\n")

    # ==================================================
    # LE VOTE MAJORITAIRE ICI !
    # ==================================================
    from collections import Counter
    
    print("==================================================")
    print("🗳️ FINAL EVALUATION: MAJORITY VOTE (DOCUMENT LEVEL)")
    print("==================================================")
    
    doc_predictions = {}
    doc_true_labels = {}

    for pred, true_label, doc_id in zip(all_preds, all_labels, test_ds.chunk_doc_ids):
        if doc_id not in doc_predictions:
            doc_predictions[doc_id] = []
            doc_true_labels[doc_id] = true_label
        
        doc_predictions[doc_id].append(pred)

    final_doc_preds = []
    final_doc_labels = []

    for doc_id, preds in doc_predictions.items():
        majority_pred = Counter(preds).most_common(1)[0][0]
        final_doc_preds.append(majority_pred)
        final_doc_labels.append(doc_true_labels[doc_id])

    correct_docs = sum(1 for p, t in zip(final_doc_preds, final_doc_labels) if p == t)
    doc_accuracy = correct_docs / len(final_doc_preds)
    doc_mae = sum(abs(p - t) for p, t in zip(final_doc_preds, final_doc_labels)) / len(final_doc_preds)

    print(f"📄 Total Documents Evaluated: {len(final_doc_preds)} texts")
    print(f"🎯 Document Accuracy (Vote) : {doc_accuracy * 100:.2f}%")
    print(f"📏 Document MAE             : {doc_mae:.2f} decades\n")

if __name__ == "__main__":
    main()