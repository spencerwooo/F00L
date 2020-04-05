"""
Server Chan Notifications: Push notifications to WeChat upon training complete.

! NOTE: You won't need this file unless you know what you are doing.
! You can customize your own push notification by acquiring a token at: https://sc.ftqq.com
! This script requires: https://github.com/LiJinyao/BIT-campus-network-CLI

* Prerequisites: setting environment variables
  - `BIT_ACNT`: your login account
  - `BIT_SCRT`: your login password
* CLI USAGE: python notify.py -b <BIT.js> -t <title> -m <message>
"""

import getopt
import os
import sys
import urllib.parse
import urllib.request

# get your own server chan token at: https://sc.ftqq.com
server_chan_token = 'SCU51420Tc53c54655f0a9ffe3d66789be07a51af5cda6de0e572a.'


def notify_server_chan(msg_title, msg_desp):
  msg_title = urllib.parse.quote_plus(msg_title)
  msg_desp = urllib.parse.quote_plus(msg_desp)
  url = 'https://sc.ftqq.com/{}send?text={}&desp={}'.format(server_chan_token, msg_title, msg_desp)

  f = urllib.request.urlopen(url)
  print('[Server Chan]', f.read().decode('utf-8'))


def bit_handler(action, bitjs, username, password):
  if action == 'login':
    cmd = 'node {} {} {} {}'.format(bitjs, action, username, password)
  if action == 'logout':
    cmd = 'node {} {} {}'.format(bitjs, action, username)
  stream = os.popen(cmd)
  output = stream.read()
  return output


def parse_args(argv):
  # default location
  bitjs = 'BIT.js'
  # default messages
  title = 'Congrats! Task complete.'
  msg = 'Task successfully complete. Login to check outputs.'

  try:
    opts, args = getopt.getopt(argv, 'hb:t:m:', ['bit=', 'title=', 'msg='])
  except getopt.GetoptError:
    print('[NOTIFY] USAGE: python notify.py -b <BIT.js> -t <title> -m <message>')
    sys.exit(2)

  for opt, arg in opts:
    if opt == '-h':
      print('[NOTIFY] USAGE: python notify.py -b <BIT.js> -t <title> -m <message>')
      sys.exit()
    if opt in ('-b', '--bitjs'):
      bitjs = arg
    if opt in ('-t', '--title'):
      title = arg
    if opt in ('-m', '--msg', '--message'):
      msg = arg

  return bitjs, title, msg


def main(argv):
  bitjs, title, msg = parse_args(argv)
  if (len(argv) == 0):
    print('[NOTIFY] Default message detected.\n[NOTIFY] Title: {}\n[NOTIFY] Message: {}'.format(title, msg))
  else:
    print('[NOTIFY] Parsing message...\n[NOTIFY] Title: {}\n[NOTIFY] Message: {}'.format(title, msg))

  # get env var
  BIT_ACNT = os.environ.get('BIT_ACNT')
  BIT_SCRT = os.environ.get('BIT_SCRT')
  if (BIT_ACNT != None and BIT_SCRT != None):
    print('[NOTIFY] Got login info. Logging in...')

    # send notifications
    try:
      # lin
      lin_resp = bit_handler('login', bitjs, BIT_ACNT, BIT_SCRT)
      print('[NOTIFY] Web login:\n', lin_resp)

      if ('login successfully' in lin_resp):
        # notify
        notify_server_chan(title, msg)
        print('[NOTIFY] Notification sent! Logging out...')
      else:
        print('[NOTIFY] Login failed! Aborting...')

      # lout
      lout_resp = bit_handler('logout', bitjs, BIT_ACNT, BIT_SCRT)
      print('[NOTIFY] Web logout:\n', lout_resp)

    except Exception as e:
      print('[NOTIFY] Notify failed: ', e)

  else:
    print('[NOTIFY] Get login info failed. Trying direct notification...')
    try:
      notify_server_chan(title, msg)
      print('[NOTIFY] Notification sent!')
    except Exception as e:
      print('[NOTIFY] Notify failed: ', e)


if __name__ == "__main__":
  main(sys.argv[1:])
