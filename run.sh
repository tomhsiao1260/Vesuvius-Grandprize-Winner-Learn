#!/bin/sh

# username = 'registeredusers'
# password = 'only'

echo username?
read USERNAME
echo password?
read PASSWORD

rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230509182749/20230509182749_mask.png ./train_scrolls/20230509182749/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8
rclone copy :http:/full-scrolls/Scroll1.volpkg/paths/20230509182749/layers/ ./train_scrolls/20230509182749/layers/ --http-url http://$USERNAME:$PASSWORD@dl.ash2txt.org/ --progress --multi-thread-streams=8 --transfers=8

