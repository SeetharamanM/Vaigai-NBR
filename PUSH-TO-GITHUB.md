# Push this project to GitHub

Your repo is already configured with remote **VNBR** → `https://github.com/SeetharamanM/VNBR.git`.

## If you see "index.lock" or "Another git process" error

1. Close any Git GUI, VS Code/Cursor source control, or other app using this folder.
2. Delete the lock file (in PowerShell from this folder):
   ```powershell
   Remove-Item -Force .git\index.lock -ErrorAction SilentlyContinue
   ```

## Steps to push

Run these in PowerShell from the project folder  
`E:\OneDrive\Cursor\RCC RW dashboard`:

```powershell
# 1. Add all project files (.__pycache__ is ignored via .gitignore)
git add .

# 2. First commit
git commit -m "Initial commit: RCC RW dashboard and overlap/gap app"

# 3. Push to GitHub (use remote name VNBR, branch main)
git push -u VNBR main
```

If the remote repo is empty, this creates `main` on GitHub. If GitHub already has content (e.g. README), you may need to pull first:

```powershell
git pull VNBR main --allow-unrelated-histories
git push -u VNBR main
```

**Note:** Pushing will prompt for your GitHub username and password. Use a **Personal Access Token** instead of your account password (GitHub no longer accepts account passwords for git).

- Create a token: GitHub → Settings → Developer settings → Personal access tokens.
- When prompted for password, paste the token.
