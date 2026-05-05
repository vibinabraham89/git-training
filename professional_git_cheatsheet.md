# 🚀 Professional Git + GitLab Workflow Cheat Sheet

This document explains a **real-world Git workflow** used in teams with GitLab.

---

# 🧠 1. Core Mental Model

Git operates on three layers:

Working Directory → Staging Area → Repository (Commits)

- Working Directory → where you edit code
- Staging Area → what you plan to commit
- Repository → saved snapshots (history)

---

## 🔹 Remote Concept

origin → https://gitlab.com/your-repo.git

👉 `origin` is just a name pointing to your remote repository

---

# 🔁 2. Standard Workflow

1. Sync main branch  
2. Create feature branch  
3. Make changes  
4. Stage + commit  
5. Push feature branch  
6. Create Merge Request  
7. Merge into main  
8. Sync local main  
9. Delete feature branch  

---

# ⚙️ 3. Commands + Explanation

---

## 🔹 Check status

git status  

👉 Shows:
Working directory vs Git tracking

---

## 🔹 Stage changes

git add app.py  

👉 Moves:
Working Directory → Staging Area

---

## 🔹 Commit changes

git commit -m "chore: initial commit with basic user listing"

👉 Creates snapshot of staged files

---

## 🔹 Push to remote

git push origin main  

👉 Sends commits to GitLab  
❗ Does NOT merge branches

---

## 🔹 Update main branch

git checkout main  
git pull origin main  

👉 Fetch + merge into local main

---

# 🌱 4. Feature Development Workflow

---

## 🔹 Create feature branch

git checkout -b feature/filter-users-by-age  

👉 Creates local branch from current commit

---

## 🔹 Check changes

git diff  

👉 Shows:
Working Directory vs Staging Area

---

## 🔹 Commit feature

git add .  
git commit -m "feat: add filtering of users by minimum age"

---

## 🔹 Commit types

- feat → new functionality  
- fix → bug fix  
- chore → setup  

---

## 🔹 Push feature branch

git push origin feature/filter-users-by-age  

👉 Creates branch in GitLab

---

## 🔹 Create Merge Request

👉 Feature → main  
👉 Add title + description  
👉 Request review  

---

# 🔀 5. Merge Request Options

---

## ✔ Delete source branch

👉 Removes feature branch after merge

---

## ✔ Squash commits

Before:
- feat: add logic  
- fix: typo  
- refactor: cleanup  

After:
- feat: add filtering  

👉 Clean history

---

# 🔄 6. After Merge

---

## 🔹 Sync local main

git checkout main  
git pull origin main  

---

## 🔹 Delete feature branch

git branch -d feature/filter-users-by-age  

---

# 🔀 7. Merge vs Rebase

---

## 🟡 MERGE

git checkout main  
git merge feature/filter-users-by-age  

### Internal behavior:

main:     A ---- B ---- C ---- F ---- G -------- M  
                 \                          /  
feature:          D ---- E ------------------  

👉 M = merge commit (two parents)

---

### When to use merge

✔ Merging into main  
✔ Preserve history  
✔ Team merge workflow  

---

## 🟡 REBASE

git checkout feature/filter-users-by-age  
git rebase main  

### Internal behavior:

Original:
D ---- E  

After:
main:     A ---- B ---- C ---- F ---- G  
                                      \  
feature:                               D' ---- E'  

👉 Commits are recreated

---

### When to use rebase

✔ Update feature branch  
✔ Clean history  
✔ Local work  

---

# 🔄 8. Updating Feature Branch

---

## Option 1 — Merge

git checkout feature/filter-users-by-age  
git fetch origin  
git merge origin/main  

👉 Same as:
git pull origin main  

---

## Option 2 — Rebase (Preferred)

git checkout feature/filter-users-by-age  
git fetch origin  
git rebase origin/main  

---

# 🧠 9. Key Concepts

---

✔ git push  
- Sends commits to remote  
- Does NOT merge branches  

✔ git pull origin main  
- Merges origin/main into CURRENT branch  
- Does NOT update local main unless you're on it  

✔ Branch creation  
git checkout -b feature/...  
- No merge  
- Just creates pointer  

✔ Git depends on current branch (HEAD)

---

# 🔁 10. End-to-End Example

git checkout main  
git pull origin main  

git checkout -b feature/filter-users-by-age  

git add .  
git commit -m "feat: add filtering logic"  

git push origin feature/filter-users-by-age  

👉 Create Merge Request in GitLab  

git checkout main  
git pull origin main  

git branch -d feature/filter-users-by-age  

---

# 🧠 Final Mental Model

Edit → Stage → Commit → Push → MR → Merge → Sync

---

# 🚀 Next Steps

- Learn interactive rebase (git rebase -i)  
- Practice conflict resolution  
- Explore GitLab CI/CD  
