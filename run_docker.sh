#!/usr/bin/expect -f
spawn su
expect "密码："
send "ycq250199\r"
expect "#"
send "docker exec stgat python test_gnn_offset.py\r"
expect "#"
send "exit\r"
expect eof
