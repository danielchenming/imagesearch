echo "(1/2)开始重建图片索引"
cd /opt/sis || exit
python3 offline.py > /dev/null 2>&1 &
show_progress $!
wait $!
echo "(1/2)重建图片索引完成"

echo "(2/2)开始重建图片索引"
cd /opt/sis || exit
python3 server.py > /dev/null 2>&1 &
show_progress $!
wait $!
echo "(2/2)重建图片索引完成"
