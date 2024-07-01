import json
import struct
import socket
import traceback
import datetime

def sendJson(request, jsonData):
    data = json.dumps(jsonData).encode()
    request.send(struct.pack('i', len(data)))
    request.sendall(data)


def recvJson(request):
    data = request.recv(4)
    length = struct.unpack('i', data)[0]
    data = request.recv(length).decode()
    while len(data) != length:
        data = data + request.recv(length - len(data)).decode()
    data = json.loads(data)
    return data

class Agent():

    def __init__(self, room_id, room_number, name, game_number, server, port, bots=None, verbose=False):
        if bots is None:
            bots = []
        self.info = dict(info='connect', room_id=room_id, name=name, room_number=room_number, game_number=game_number, bots=bots)
        self.server = server
        self.port = port
        self.game_counter = 0
        self.verbose = verbose
        self.cur_id = 0
        self.turn = 0
        if self.verbose:
            print(self.info)

    def output(self, data):
        if data["action_history"] == [[]]:
            print("=======================  game {} start ============================".format(self.game_counter))
            print("your card: {}".format(data["private_card"]))
        
        if len(data["action_history"][-1]) == 0:
            if len(data["action_history"]) > 1:
                action = data["action_history"][-2][-1]
                if action['position'] == data['position']:
                    print("you {}".format(action['action']))
                else:
                    print("opp {}".format(action['action']))

            print("----------round {} ------------".format(len(data["action_history"])))
            print("public card: {}".format(data["public_card"]))

        else:
            action = data["action_history"][-1][-1]
            if action['position'] == data['position']:
                print("you {}".format(action['action']))
            else:
                print("opp {}".format(action['action']))

    def output_result(self, data):
        print("------------------------")
        print(datetime.datetime.now(), 'win money: {},\tyour card: {},\topp card: {},\t\tpublic card: {}'.format(data['players'][self.position]['win_money'], data['player_card'][self.position], data['player_card'][1 - self.position], data['public_card'])) 

    def run(self):
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((self.server, self.port))
        try:
            sendJson(client, self.info)
            while True:
                data = recvJson(client)
                if data['info'] == 'state':
                    self.position = data['position']
                    if self.verbose:
                        self.output(data)
                    if data['position'] == data['action_position']:
                        action = self.get_action(data)
                        self.turn += 1
                        sendJson(client, {'action': action, 'info': 'action'})
                elif data['info'] == 'result':
                    if self.verbose:
                        self.output_result(data)
                    sendJson(client, {'info': 'ready', 'status': 'start'})
                    self.game_counter += 1
                    self.turn = 0
                else:
                    break
        except Exception as e:
            traceback.print_exc()
        finally:
            client.close()
    
    def get_action(self, data):
        return NotImplemented