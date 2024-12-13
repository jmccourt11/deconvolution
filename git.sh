#!/bin/bash

git status
git add . --all
read -sp "Enter commit message: " commit
echo $commit
git commit -m $commit 
token=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
<<<<<<< HEAD
git push
=======
git push 
>>>>>>> e2c5a0a9de4800a1657e831e020710cd35a4dc12
