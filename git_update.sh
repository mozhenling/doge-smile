#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'add link; be concise'

git push origin master

echo '------- update complete --------'