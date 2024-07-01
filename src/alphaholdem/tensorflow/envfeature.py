# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:51:36 2021

@author: EnminZhao
"""

import numpy as np
import random 
import copy
from .comparehands import compare_hands
#e.g. hands = [['CT', 'ST', 'H9', 'B9', 'C2', 'C8', 'C7'], ['CJ', 'SJ', 'H9', 'B9', 'C2', 'C8', 'C7'], ['CT', 'ST', 'H9', 'B9', 'C2', 'C8', 'C7']]
class holdem2ENV:
    def __init__(self):
        self.player1hand=np.zeros((4,13))
        self.player2hand=np.zeros((4,13))
        #self.publichand=np.zeros((4,13,4))
        self.bet=150
        #self.action_number=0
        self.bet_num=0
        self.player1_bet=50
        self.player2_bet=100
        self.action_history_info=[]
        self.player_id=0
        self.maxpot=20000
        self.player_num=2
        self.fold=[0,0]
        self.allin_true=False
        self.hands_num=52
        self.action_history=[[],[],[],[]]
        self.RANK_TO_STRING = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
                               7: "7", 8: "8", 9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
        self.STRING_TO_RANK = {v:k for k, v in self.RANK_TO_STRING.items()}
        self.RANK_LOOKUP = "23456789TJQKA"
        self.SUIT_LOOKUP = "scdh"
        self.pid=0
    
    def reset(self):
        self.player1hand=np.zeros((4,13))
        self.player2hand=np.zeros((4,13))
        #self.publichand0=np.zeros((4,13))
        self.publichand1=np.zeros((4,13))
        self.publichand2=np.zeros((4,13))
        self.publichand3=np.zeros((4,13))
        self.action_record0=np.zeros((4,9,13))
        self.action_record1=np.zeros((4,9,13))
        self.action_record2=np.zeros((4,9,13))
        self.action_record3=np.zeros((4,9,13))
        self.pid=0
        self.turnbet=np.zeros(4)        
        self.action_number=np.zeros(4)
        self.turn=0
        self.bet_num=0
        self.bet=150
        self.turn=0
        
        self.player_num=2
        self.player1_bet=50
        self.player2_bet=100
    
        self.action_history_info=[]
        self.player_id=0
        self.maxpot=20000
        self.fold=[0,0]
        self.allin_true=False
        self.hands_num=52
        self.done=False
        self.actions=['fold','check','call','bet_halfpot','bet_3quarpot','bet_pot','bet_15pot','bet_2pot','allin']
        self.action_history=[[],[],[],[]]
        self.RANK_TO_STRING = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
                               7: "7", 8: "8", 9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
        self.STRING_TO_RANK = {v:k for k, v in self.RANK_TO_STRING.items()}
        self.RANK_LOOKUP = "23456789TJQKA"
        self.SUIT_LOOKUP = "scdh"

        self.allhands=list(range(self.hands_num))
        
        self.publichand_comp=[]
        self.player1hand_comp=[]
        self.player2hand_comp=[]

      
        self.rewards=np.zeros(2)
        legal_action=np.ones(9)
        legal_action[1]=0
        legal_action[8]=0
        self.legal_action=legal_action
        self.envhands=np.zeros((4,13,6))
        obs=np.zeros(312+1872+10)
        obs[0:312]=self.envhands.reshape(312)
        obs[312:312+468]=self.action_record0.reshape(468)
        obs[312+468:312+468*2]=self.action_record1.reshape(468)
        obs[312+468*2:312+468*3]=self.action_record2.reshape(468)
        obs[312+468*3:312+468*4]=self.action_record3.reshape(468)

        obs[1872+312:1872+312+9]=legal_action
        obs[1872+312+9]=self.player_id
        obs1=obs.copy()
        self.obs=obs

        return obs1
        
    def judgeallin(self,player_bet):
        if player_bet>self.maxpot-1:
            player_bet=self.maxpot
            self.allin_true=True
        return player_bet
    
    def getobs(self):
        obs=self.obs
        obs1=obs.copy()
        return obs1

    def step(self,action):
        '''
        Get env step
        Returns:
            (list): the best five cards among the seven cards of a player
        action: fold check call bet 0.25 0.5 0.75 1 2  allin
        '''
        #self.turn=int(self.turn)
        if self.turn==0:
            self.action_record0[self.player_id,action,int(self.action_number[self.turn])]=1
            self.action_record0[2,action,int(self.action_number[self.turn])]=1
            self.action_number[self.turn]=self.action_number[self.turn]+1
        elif self.turn==1:
            self.action_record1[self.player_id,action,int(self.action_number[self.turn])]=1
            self.action_record1[2,action,int(self.action_number[self.turn])]=1
            self.action_number[self.turn]=self.action_number[self.turn]+1
        elif self.turn==2:
            self.action_record2[self.player_id,action,int(self.action_number[self.turn])]=1
            self.action_record2[2,action,int(self.action_number[self.turn])]=1
            self.action_number[self.turn]=self.action_number[self.turn]+1
        else:
            self.action_record3[self.player_id,action,int(self.action_number[self.turn])]=1
            self.action_record3[2,action,int(self.action_number[self.turn])]=1
            self.action_number[self.turn]=self.action_number[self.turn]+1
            
        if self.legal_action[action]==0 and action==8:
            raise Error
        action=self.actions[action]
       
        if self.player_id==0:
            if action=='fold':
                self.player_num=self.player_num-1
                self.fold[0]=1
            elif action=='check':
                self.player1_bet=self.player1_bet if self.player1_bet>99 else 100
            elif action=='call':
                self.player1_bet=max(self.player1_bet,self.player2_bet)
            elif action=='bet_halfpot':
                self.player1_bet=2*max(self.player1_bet,self.player2_bet)
            elif action=='bet_3quarpot':
                self.player1_bet=2.5*max(self.player1_bet,self.player2_bet)
            elif action=='bet_pot':
                self.player1_bet=3*max(self.player1_bet,self.player2_bet)
            elif action=='bet_15pot':
                self.player1_bet=4*max(self.player1_bet,self.player2_bet)
            elif action=='bet_2pot':
                self.player1_bet=5*max(self.player1_bet,self.player2_bet)

            else:
                self.player1_bet=self.maxpot
                self.player1_bet=self.judgeallin(self.player1_bet)
            self.bet=self.player1_bet+self.player2_bet
            self.player_id=1
           

        else:
            if action=='fold':
                self.fold[1]=1
                self.player_num=self.player_num-1
            elif action=='check':
                self.player2_bet=self.player2_bet
            elif action=='call':
                self.player2_bet=max(self.player1_bet,self.player2_bet)

            elif action=='bet_halfpot':
                self.player2_bet=2*max(self.player1_bet,self.player2_bet)
            elif action=='bet_3quarpot':
                self.player2_bet=2.5*max(self.player1_bet,self.player2_bet)
            elif action=='bet_pot':
                self.player2_bet=3*max(self.player1_bet,self.player2_bet)
            elif action=='bet_15pot':
                self.player2_bet=4*max(self.player1_bet,self.player2_bet)
            elif action=='bet_2pot':
                self.player2_bet=5*max(self.player1_bet,self.player2_bet)

            else:
                self.player2_bet=self.maxpot
                self.player2_bet=self.judgeallin(self.player2_bet)
            self.bet=self.player1_bet+self.player2_bet
            self.player_id=0
                    
                    
        self.bet=self.player1_bet+self.player2_bet
        self.action_history[self.turn].append(action[0])
        self.actions_judge()
        self.judgeturndone()
        if self.turn==4:
            legal_action=np.zeros(9)
        else:
            legal_action=self.getlegal_action()
        
        
        obs=np.zeros(312+1872+10)
        obs[0:312]=self.envhands.reshape(312)
        obs[312:312+468]=self.action_record0.reshape(468)
        obs[312+468:312+468*2]=self.action_record1.reshape(468)
        obs[312+468*2:312+468*3]=self.action_record2.reshape(468)
        obs[312+468*3:312+468*4]=self.action_record3.reshape(468)
        obs[1872+312:1872+312+9]=legal_action
        obs[1872+312+9]=self.player_id
        
        obs1=obs.copy()
        self.obs=obs

        rewards=self.rewards/100
        rewards1=rewards.copy()
        done=copy.deepcopy(self.done)
        action_history= copy.deepcopy(self.action_history)

        
        return obs1
    
    def getlegal_action(self):
        legal_action=np.ones(9)
        if len(self.action_history[self.turn])==0:
            
            legal_action[2]=0
        if len(self.action_history[self.turn])==1:
            if self.action_history[self.turn][-1]=='c':
                legal_action[2]=0
            else:
                legal_action[1]=0
        else:
            for i in range(len(self.action_history[self.turn])):
                if self.action_history[self.turn][-i]=='b':
                    legal_action[1]=0
                if self.action_history[self.turn][-i]=='a':
                    legal_action[:]=0
                    legal_action[2]=1
                    legal_action[0]=1
                
            
        if self.bet_num[self.turn]>4:
            legal_action[:]=0
            legal_action[2]=1
            legal_action[0]=1
 

        if 2*max(self.player1_bet,self.player2_bet)>19999:
            legal_action[3:8]=0

        if 2.5*max(self.player1_bet,self.player2_bet)>19999:
            legal_action[4:8]=0
        if 3*max(self.player1_bet,self.player2_bet)>19999:
            legal_action[5:8]=0
        if 4*max(self.player1_bet,self.player2_bet)>19999:
            legal_action[6:8]=0
        if 5*max(self.player1_bet,self.player2_bet)>19999:
            legal_action[7]=0

        if 3*max(self.player1_bet,self.player2_bet)<20000:
            legal_action[8]=0
            
        self.legal_action=legal_action
        return legal_action
            
        
    def judgeturndone(self):
        if len(self.action_history[self.turn])>1 and self.action_history[self.turn][-1]=='c':
            if self.pid==0:
                self.turnbet[self.turn]=self.player1_bet
            else:
                self.turnbet[self.turn]=self.player2_bet
            if self.player1_bet==self.player2_bet:
                self.turn=self.turn+1            
                self.player_id=1
            else:
                print("self.player1_bet==self.player2_bet",self.player1_bet,self.player2_bet)
                return Error

    def judgedone(self):
        
        if self.player_num==1:
            a=1
            self.done=True
            self.rewards[0]=-self.player1_bet
            self.rewards[1]=-self.player2_bet
            for i in range(2):
                if self.fold[i]==0:
                    self.rewards[i]=0
        else:
            if self.turn==3:
                if len(self.action_history[3])>1 and self.action_history[3][-1]=='c':
                    self.done=True
                    self.rewards[0]=-self.player1_bet
                    self.rewards[1]=-self.player2_bet
                    self.player1hand_comp=self.player1hand_comp+self.publichand_comp
                    self.player2hand_comp=self.player2hand_comp+self.publichand_comp
                    result=compare_hands([self.player1hand_comp,self.player2hand_comp])
             
                    if sum(result)==1:
                        a=1
                        if result[0]==1:
                            self.rewards[0]=0
                        else:
                            self.rewards[1]=0
                    else:
                        a=2
                        self.rewards[0]=0
                        self.rewards[1]=0

                

        if self.done==True:

            if a==1:
                for i in range(2):
                    if self.rewards[i]==0:
                        self.rewards[i]=-sum(self.rewards)

                    
    def actions_judge(self):
        self.bet_num=np.zeros(4)
        for i in range(len(self.action_history[self.turn])):
            if self.action_history[self.turn][i]=='b' or self.action_history[self.turn][i]=='a':
                self.bet_num[self.turn]=self.bet_num[self.turn]+1


        

    def getbetaction(self,ratio):
        if ratio<0.625 and ratio>0:
            action=3
        elif ratio>0.625 and ratio<0.875:
            action=4
        elif ratio>0.875 and ratio<1.25:
            action=5
        elif ratio>1.25 and ratio<1.75:
            action==6
        elif ratio>1.75 and ratio<2.01:
            action=7
        else:
            action==0
        return action
    def perform_env_and_oppoaction(self,info):
        
        self.pid=info['position']
        hole_cards=info['private_card']
        board_cards=info['public_card']       
        self.getplayerhands(hole_cards)
        self.getboardhands(board_cards)
        
        action_history_get=info['action_history']
        action_history_old=self.action_history_info 
        action_history_new=[]
        for i in range(len(action_history_get)):
            for j in range(len(action_history_get[i])):
                action_history_new.append(action_history_get[i][j])
                
                
        for i in range(len(action_history_old),len(action_history_new)):
            if action_history_new[i]==[]:
                action_his=[]
            else:
                action_his=action_history_new[i]
            if action_his!=[] and action_his['position']==1-self.pid:
                action_this_turn=action_his['action']
                if action_this_turn=='fold':
                    action=0
                elif action_this_turn=='check':
                    action=1
                elif action_this_turn=='call':
                    action=2
                else:
                    if info['players'][1-self.pid]['money_left']==0:
                        action=8
                    else:
                        ratio=0.5*(20000-info['players'][1-self.pid]['money_left']-max(self.player1_bet,self.player2_bet))/max(self.player1_bet,self.player2_bet)
                        action=self.getbetaction(ratio)
                print("player&action",self.player_id,action)
                obs=self.step(action)
                print("playerafteraction",self.player_id)
                
                
        self.action_history_info=action_history_new
        obs=self.getobs()
        return obs


    def getplayerhands(self,hole_cards):
        hands=np.zeros((4,13))
        
        for i in range(13):
            for j in range(4):
                if hole_cards[0][0]==self.RANK_LOOKUP[i] and hole_cards[0][1]==self.SUIT_LOOKUP[j]:
                    hands[j,i]=1
                if hole_cards[1][0]==self.RANK_LOOKUP[i] and hole_cards[1][1]==self.SUIT_LOOKUP[j]:
                    hands[j,i]=1
        
        self.envhands[:,:,0]=hands
    
    def getboardhands(self,board_cards):
        hands1=np.zeros((4,13))
        hands2=np.zeros((4,13))
        hands3=np.zeros((4,13))
        lengthborad=len(board_cards)
        #print('lengthborad',lengthborad)
        if lengthborad==3:
            hands2=np.zeros((4,13))
            hands3=np.zeros((4,13))
            for i in range(13):
                for j in range(4):
                    if board_cards[0][0]==self.RANK_LOOKUP[i] and board_cards[0][1]==self.SUIT_LOOKUP[j]:
                        hands1[j,i]=1
                    if board_cards[1][0]==self.RANK_LOOKUP[i] and board_cards[1][1]==self.SUIT_LOOKUP[j]:
                        hands1[j,i]=1
                    if board_cards[2][0]==self.RANK_LOOKUP[i] and board_cards[2][1]==self.SUIT_LOOKUP[j]:
                        hands1[j,i]=1
        elif lengthborad==4:
            for i in range(13):
                for j in range(4):
                    if board_cards[0][0]==self.RANK_LOOKUP[i] and board_cards[0][1]==self.SUIT_LOOKUP[j]:
                        hands1[j,i]=1
                    if board_cards[1][0]==self.RANK_LOOKUP[i] and board_cards[1][1]==self.SUIT_LOOKUP[j]:
                        hands1[j,i]=1
                    if board_cards[2][0]==self.RANK_LOOKUP[i] and board_cards[2][1]==self.SUIT_LOOKUP[j]:
                        hands1[j,i]=1
                    if board_cards[3][0]==self.RANK_LOOKUP[i] and board_cards[3][1]==self.SUIT_LOOKUP[j]:
                        hands2[j,i]=1
                        
        elif lengthborad==5:
            for i in range(13):
                for j in range(4):
                    if board_cards[0][0]==self.RANK_LOOKUP[i] and board_cards[0][1]==self.SUIT_LOOKUP[j]:
                        hands1[j,i]=1
                    if board_cards[1][0]==self.RANK_LOOKUP[i] and board_cards[1][1]==self.SUIT_LOOKUP[j]:
                        hands1[j,i]=1
                    if board_cards[2][0]==self.RANK_LOOKUP[i] and board_cards[2][1]==self.SUIT_LOOKUP[j]:
                        hands1[j,i]=1
                    if board_cards[3][0]==self.RANK_LOOKUP[i] and board_cards[3][1]==self.SUIT_LOOKUP[j]:
                        hands2[j,i]=1
                    if board_cards[4][0]==self.RANK_LOOKUP[i] and board_cards[4][1]==self.SUIT_LOOKUP[j]:
                        hands3[j,i]=1
        else:
            hands1=np.zeros((4,13))
            hands2=np.zeros((4,13))
            hands3=np.zeros((4,13))
                        
        
        self.envhands[:,0:13,1]=hands1
        self.envhands[:,0:13,2]=hands2
        self.envhands[:,0:13,3]=hands3
        self.envhands[:,0:13,4]=hands1+hands2+hands3
        self.envhands[:,0:13,5]=hands1+hands2+hands3+self.envhands[:,:,0]      
        self.obs[0:312]=self.envhands.reshape(312)
