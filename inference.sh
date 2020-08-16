if [ -z $1 ] ; then
  echo "First parameter needed!" && exit 1;
fi

# curl http://127.0.0.1:8080/predictions/cartoonize -T "$1"