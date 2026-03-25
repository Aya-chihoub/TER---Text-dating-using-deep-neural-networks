import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Importation des modules de votre projet
from src.utils.config import get_default_configs
from src.data.parser import load_corpus
from src.data.split import stratified_split
from src.data.dataset import build_datasets
from src.models.cnn import AgeCNN

def main():
    # 1. Charger la configuration
    print("Chargement des configurations...")
    data_cfg, feat_cfg, model_cfg, train_cfg = get_default_configs()
    
    # Configuration du device (détecte automatiquement ton GPU si CUDA est dispo, sinon CPU)
    device = torch.device("cuda" if torch.cuda.is_available() and train_cfg.device == "cuda" else "cpu")
    print(f"Entraînement lancé sur : {device}")

    # 2. Charger le corpus complet en mémoire
    print("\nChargement du corpus...")
    entries = load_corpus(data_cfg.corpus_dir)
    print(f"Total des textes trouvés dans le dossier : {len(entries)}")
    
    # ---------------------------------------------------------
    # MODE TEST : On ne garde que 100 textes au hasard
    # ---------------------------------------------------------
    nombre_textes_test = 100
    if len(entries) > nombre_textes_test:
        # On utilise random.seed pour avoir toujours les mêmes 100 textes à chaque test
        random.seed(data_cfg.random_seed) 
        entries = random.sample(entries, nombre_textes_test)
    print(f"⚠️ MODE TEST ACTIVÉ : Utilisation de {len(entries)} textes uniquement.")
    # ---------------------------------------------------------

    # 3. Créer le split (Train/Test)
    train_entries, test_entries = stratified_split(
        entries, 
        n_test_per_age=data_cfg.n_test_per_age, 
        random_seed=data_cfg.random_seed
    )

    # 4. Construire les datasets (l'extraction des 14 features se fait ici)
    print("\nConstruction des Datasets (extraction des features en cours)...")
    train_ds, test_ds = build_datasets(
        train_entries, 
        test_entries,
        sequence_length=data_cfg.sequence_length,
        train_stride=data_cfg.stride
    )

    # 5. Créer les DataLoaders
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False)

    # 6. Initialiser le Modèle, la Loss et l'Optimiseur
    model = AgeCNN(model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=train_cfg.learning_rate, 
        weight_decay=train_cfg.weight_decay
    )
    
    # Scheduler pour réduire le LR si l'apprentissage stagne
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=train_cfg.lr_scheduler_factor, 
        patience=train_cfg.lr_scheduler_patience
    )

    # Préparation pour la sauvegarde (Early Stopping)
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # 7. Boucle d'entraînement
    print("\nDébut de l'entraînement...")
    for epoch in range(train_cfg.num_epochs):
        
        # --- PHASE D'ENTRAÎNEMENT ---
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for features, labels in train_loader:
            # On envoie les tenseurs sur la carte graphique (ou le CPU)
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(logits, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        avg_train_loss = train_loss / total_train if total_train > 0 else 0
        train_acc = 100 * correct_train / total_train if total_train > 0 else 0
        
        # --- PHASE DE VALIDATION ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(logits, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        avg_val_loss = val_loss / total_val if total_val > 0 else 0
        val_acc = 100 * correct_val / total_val if total_val > 0 else 0
        
        # Mise à jour du scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{train_cfg.num_epochs}] "
              f"| Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # --- EARLY STOPPING & CHECKPOINT ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(train_cfg.checkpoint_dir, "best_model.pth"))
            print("  -> Nouveau meilleur modèle sauvegardé !")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg.patience:
                print(f"\nArrêt anticipé (Early Stopping) déclenché à l'epoch {epoch+1}")
                break

if __name__ == "__main__":
    main()