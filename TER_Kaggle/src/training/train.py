import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import your project modules
from src.utils.config import get_default_configs
from src.data.parser import load_corpus
from src.data.split import split_train_val_test
from src.data.dataset import build_datasets, TextAgeDataset
from src.models.cnn import AgeCNN
from src.data.features import FormFeatureExtractor

def main():
    import builtins

    def print(*args, **kwargs):
        kwargs.setdefault("flush", True)
        builtins.print(*args, **kwargs)

    print("Loading configurations...")
    data_cfg, feat_cfg, model_cfg, train_cfg = get_default_configs()

    print(f"\n{'=' * 60}")
    print("RUN CONFIGURATION (features + training + model)")
    print(f"{'=' * 60}")
    print(f"Experiment name:      {train_cfg.experiment_name}")
    print(f"Active feature_dim:   {feat_cfg.feature_dim}  (model input channels: {model_cfg.feature_dim})")
    print("Feature flags:")
    print(f"  use_freq_count:        {feat_cfg.use_freq_count}")
    print(f"  use_log_freq:          {feat_cfg.use_log_freq}")
    print(f"  use_global_freq:       {feat_cfg.use_global_freq}")
    print(f"  use_char_count:        {feat_cfg.use_char_count}")
    print(f"  use_punctuation_type:  {feat_cfg.use_punctuation_type}")
    print(f"  use_pos_tag:           {feat_cfg.use_pos_tag}")
    print("Training:")
    print(f"  dropout (head):        {model_cfg.dropout}")
    print(f"  learning_rate:         {train_cfg.learning_rate}")
    print(f"  weight_decay:          {train_cfg.weight_decay}")
    print(f"  label_smoothing:       {train_cfg.label_smoothing}")
    print("Model:")
    print(f"  branches (kernels):   {len(model_cfg.kernel_sizes)}  {model_cfg.kernel_sizes}")
    print(f"  num_filters:          {model_cfg.num_filters}")
    print(f"  num_conv_layers/branch:{model_cfg.num_conv_layers}")
    print("Normalization:        global StandardScaler on train rows (dataset.py)")
    if feat_cfg.feature_dim != model_cfg.feature_dim:
        raise ValueError(
            f"feature_dim mismatch: FeatureConfig={feat_cfg.feature_dim}, "
            f"ModelConfig={model_cfg.feature_dim}"
        )
    print(f"{'=' * 60}\n")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 SUCCESS: NVIDIA GPU detected! Training on: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ WARNING: No GPU detected by PyTorch. Falling back to CPU.")

    print("\nLoading the full corpus...")
    entries = load_corpus(data_cfg.corpus_dir)
    
    # 1. Split: 10% test, then 90% → 85% train + 15% val (stratified by age)
    train_entries, val_entries, test_entries = split_train_val_test(
        entries,
        test_size=data_cfg.test_size,
        val_size=data_cfg.val_size,
        random_seed=data_cfg.random_seed,
    )
    
    n_all = len(entries)
    print(f"\n📊 Data Distribution (test={data_cfg.test_size:.0%}, "
          f"val={data_cfg.val_size:.0%} of remainder, stratified by age):")
    print(f"   - Train : {len(train_entries)} texts ({100 * len(train_entries) / n_all:.1f}% of corpus)")
    print(f"   - Val   : {len(val_entries)} texts ({100 * len(val_entries) / n_all:.1f}%)")
    print(f"   - Test  : {len(test_entries)} texts ({100 * len(test_entries) / n_all:.1f}%)")

  
   # 2. Build datasets
    print("\nBuilding Datasets (Feature extraction in progress)...")
    
    # We just pass the entire feat_cfg object directly! Clean and simple.
    my_extractor = FormFeatureExtractor(config=feat_cfg)
    
    train_ds, val_ds, feature_scaler = build_datasets(
        train_entries,
        val_entries,
        extractor=my_extractor,
        sequence_length=data_cfg.sequence_length,
        train_stride=data_cfg.stride,
    )

    print("\nBuilding test dataset (feature extraction, same as val) ...")
    test_ds = TextAgeDataset(
        test_entries,
        extractor=my_extractor,
        sequence_length=data_cfg.sequence_length,
        stride=data_cfg.sequence_length,
        feature_scaler=feature_scaler,
        progress_tag="test windows",
    )
    print(f"  → {len(test_ds)} test samples")

    # Quick check: train rows ~standardized globally; val differs slightly (expected)
    print("\n=== Global normalization (sample windows) ===")
    x_tr = train_ds[0][0].numpy()
    x_va = val_ds[0][0].numpy()
    print(f"First train window: mean={x_tr.mean():.4f}, std={x_tr.std():.4f} (per-window, not full train set)")
    print(f"First val window:   mean={x_va.mean():.4f}, std={x_va.std():.4f}")
    print("============================================\n")

    # 3. Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False)

    # 4. Initialize Model, Loss, Optimizer
    model = AgeCNN(model_cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'=' * 60}")
    print("MODEL ARCHITECTURE SUMMARY")
    print(f"{'=' * 60}")
    print(f"Input features: {model_cfg.feature_dim}")
    print(f"Branches: {len(model_cfg.kernel_sizes)} (kernels: {model_cfg.kernel_sizes})")
    print(f"Filters per branch: {model_cfg.num_filters}, conv blocks/branch: {model_cfg.num_conv_layers}")
    print(f"Head: Dropout → Linear({model_cfg.num_filters * len(model_cfg.kernel_sizes)} → {model_cfg.num_classes})")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'=' * 60}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)
    
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
    disable_batch_bars = os.environ.get("TER_EPOCH_ONLY", "").lower() in ("1", "true", "yes")

    # 5. Training Loop
    print("\nStarting training loop...")
    for epoch in range(train_cfg.num_epochs):
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss, correct_train, train_mae_sum, total_train = 0.0, 0, 0.0, 0
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{train_cfg.num_epochs} [Train]",
            leave=False,
            file=sys.stdout,
            disable=disable_batch_bars,
        )
        
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
        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{train_cfg.num_epochs} [Val]",
            leave=False,
            file=sys.stdout,
            disable=disable_batch_bars,
        )
        
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

        lr_now = optimizer.param_groups[0]["lr"]
        # One clear block per epoch (easy to spot in Kaggle / Jupyter logs)
        print("")
        print("=" * 72)
        print(
            f"EPOCH {epoch + 1}/{train_cfg.num_epochs}  |  "
            f"Train acc: {train_acc:.2f}%  |  Val acc: {val_acc:.2f}%"
        )
        print(
            f"  Train loss: {avg_train_loss:.4f}  |  Val loss: {avg_val_loss:.4f}  |  "
            f"Train MAE: {train_mae:.2f}  |  Val MAE: {val_mae:.2f}  |  LR: {lr_now:.2e}"
        )
        print("=" * 72)
        sys.stdout.flush()

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
        for features, labels in tqdm(test_loader, desc="Final Test", file=sys.stdout, disable=disable_batch_bars):
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