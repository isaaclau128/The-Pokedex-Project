"""
MODEL SAVING CODE FOR JUPYTER NOTEBOOK

Add this code to the end of your Pokedex.ipynb notebook after training completes.
This will save the trained model so that camera_pokemon_recognizer.py can load it.

Location: Add as a new cell at the end of the notebook
"""

# ============================================================
# PASTE THIS CODE INTO A NEW CELL AT THE END OF THE NOTEBOOK
# ============================================================

# Save the trained model
model_save_path = './pokemon_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"✓ Model saved successfully to: {model_save_path}")
print(f"✓ Model size: {os.path.getsize(model_save_path) / 1024 / 1024:.2f} MB")

# Verify the model can be loaded
print("\nVerifying model loading...")
test_model = CNNClassifier().to(device)
test_model.load_state_dict(torch.load(model_save_path, map_location=device))
test_model.eval()
print("✓ Model verification successful!")
print("\nNow you can run: python camera_pokemon_recognizer.py")

# ============================================================
# END OF CODE TO PASTE
# ============================================================
