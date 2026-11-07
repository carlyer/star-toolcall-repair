#!/usr/bin/env bash
mkdir -p "$ROOT/.cache/locks"
exec 9>"$ROOT/.cache/locks/training.lock"
flock 9
