Kaggle dataset download instructions

This project uses the Kaggle dataset `noodulz/pokemon-dataset-1000`.

Options to download:

1) Preferred: use Kaggle CLI with `~/.kaggle/kaggle.json`
   - Create a Kaggle account and go to: https://www.kaggle.com/me/account
   - Click "Create API Token" to download `kaggle.json`
   - Place it at `~/.kaggle/kaggle.json` with permissions `600`:

```bash
mkdir -p ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

   - Then run from the project root:

```bash
kaggle datasets download -d noodulz/pokemon-dataset-1000 -p pokemon-dataset-1000 --unzip
```

2) Alternative: set environment variables (CI-friendly):

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
python3 -m pip install --user kaggle
kaggle datasets download -d noodulz/pokemon-dataset-1000 -p pokemon-dataset-1000 --unzip
```

3) Manual: download from the dataset page and unzip into `pokemon-dataset-1000/`.

After download, verify the dataset with:

```bash
ls -l pokemon-dataset-1000 | head
wc -l pokemon-dataset-1000/metadata.csv
```

If you'd like, I can attempt the download now if you (a) add `~/.kaggle/kaggle.json` on this machine, or (b) paste your `KAGGLE_USERNAME` and `KAGGLE_KEY` here (not recommended publicly). Alternatively you can upload the dataset zip into the workspace and I will unzip and verify it for you.
