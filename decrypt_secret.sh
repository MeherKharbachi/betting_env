#!/bin/sh

# Decrypt the file
mkdir $HOME/secrets

n# --batch to prevent interactive command
# --yes to assume "yes" for questions
gpg --quiet --batch --yes --decrypt --passphrase="$BETTING_SECRET" \
--output $HOME/secrets/config.toml config.toml.gpg