#!/bin/bash

git status
git add . --all
read -sp "Enter commit message: " commit
echo $commit
git commit -m $commit 
token=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
git push
