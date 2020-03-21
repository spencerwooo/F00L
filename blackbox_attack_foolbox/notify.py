"""
Server Chan Notifications: Push notifications to WeChat upon training complete.

! NOTE: You won't need this file unless you know what you are doing.
! You can customize your own push notification by acquiring a token at: https://sc.ftqq.com

* Prerequisites: setting environment variables
  - `BIT_ACNT`: your login account
  - `BIT_SCRT`: your login password
* CLI USAGE: python notify.py -t <title> -m <message>
"""

import os
import sys
import getopt
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


def bit_handler(action, username, password):
  url = 'http://10.0.0.55:801/include/auth_action.php'

  data = urllib.parse.urlencode({
      'action': action,
      'username': username,
      'password': password,
      'ac_id': 8
  }).encode('utf-8')

  post_resp = urllib.request.urlopen(url=url, data=data)
  return post_resp.read().decode('utf-8')


def parse_args(argv):
  title = 'Congrats! Task complete.'
  msg = 'Task successfully complete. Login to check outputs.'

  try:
    opts, args = getopt.getopt(argv, 'ht:m:', ['title=', 'msg='])
  except getopt.GetoptError:
    print('[NOTIFY] USAGE: python notify.py -t <title> -m <message>')
    sys.exit(2)

  for opt, arg in opts:
    if opt == '-h':
      print('[NOTIFY] USAGE: python notify.py -t <title> -m <message>')
      sys.exit()
    if opt in ('-t', '--title'):
      title = arg
    if opt in ('-m', '--msg', '--message'):
      msg = arg

  return title, msg


def main(argv):
  title, msg = parse_args(argv)
  if (len(argv) == 0):
    print('[NOTIFY] Default message detected.\n[NOTIFY] Title: {}\n[NOTIFY] Message: {}'.format(title, msg))
  else:
    print('[NOTIFY] Parsing message...\n[NOTIFY] Title: {}\n[NOTIFY] Message: {}'.format(title, msg))

  # get env var
  BIT_ACNT = os.environ.get('BIT_ACNT')
  BIT_SCRT = os.environ.get('BIT_SCRT')
  if (BIT_ACNT != None and BIT_SCRT != None):
    print('[NOTIFY] Got login info. Sending notification...')

    # send notifications
    try:
      # lin
      lin_resp = bit_handler('login', BIT_ACNT, BIT_SCRT)
      print('[NOTIFY] Web login: ', lin_resp)

      if ('login_ok' in lin_resp):
        # notify
        notify_server_chan(title, msg)
        print('[NOTIFY] Notification sent! Logging out...')

      # lout
      lout_resp = bit_handler('logout', BIT_ACNT, BIT_SCRT)
      print('[NOTIFY] Web logout: ', lout_resp)

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
