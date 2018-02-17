#! /bin/bash
echo "Drake Docker container for MAC"
if [ "$#" != "2" ]; then 
  echo "Please supply two arguments: a Drake release (drake-YYYYMMDD), and a relative path to a directory to mount as /notebooks."
  exit 1
else
  IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
  echo "If something goes wrong, try chaning en0 to en1 in this script on line above"
  /usr/X11/bin/xhost + $IP &
  docker pull mit6832/drake-course:$1
  docker run -it -e DISPLAY=$IP:0 -v /tmp/.X11-unix:/tmp/.X11-unix \
              --rm -v "$(pwd)/$2":/notebooks mit6832/drake-course:$1 \
              /bin/bash -c "cd /notebooks && /bin/bash"
fi