<template>
    <div class="game-container">
        <span> 记录编号：{{ game_id }} / {{ num_games }} </span>
    </div>
    <div class="board-card-row">
        <div class="board-card-container" v-for="cardIndex in [0, 1, 2, 3, 4]">
            <div v-if="cardIndex < board_cards.length" class="board-card" :class="'board-card-' + board_cards[cardIndex][1]">
                {{ board_cards[cardIndex][0] }}
            </div>
            <div v-else class="board-card"> ? </div>
        </div>
    </div>
    <div class="observation-row">
        <div :class="'action-container action-container' + (obsIndex == current_step ? '-s' : '')"
            v-for="(observation, obsIndex) in observation_history" :key="obsIndex"
            @click="switch_obs(obsIndex)">
            <div class="action-container-info">
                <div class="action-player">
                    {{ observation.player_name[observation.player] }}
                </div>
                <div class="action-player-chips">
                    {{ observation.player_chips[observation.player] / 2 }}
                </div>
            </div>
            <div class="action-block-container">
                <div v-for="(action, actionIndex) in format_legal_actions(observation)"
                    :key="actionIndex" @click="perform_action(obsIndex, action[0])"
                    :class="'action-block action-block' + (action_history[obsIndex] == action[0] ? '-s' : '')" 
                    @mouseover="actionMouseOver(obsIndex, action[0])"
                    @mouseleave="actionMouseLeave(obsIndex, action[0])">
                    {{ action[1] }}
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">

import { useDeepStackStore } from '@/stores/deepstack'
import { storeToRefs } from 'pinia'
const deepstackStore = useDeepStackStore()

const {
    action_history,
    policy_history,
    observation_history,
    board_cards,
    current_step,
    highlight_action,
    num_games,
    game_id,
} = storeToRefs(deepstackStore)

function actionMouseOver(obsIndex: number, actionIndex: number) {
    if (obsIndex == deepstackStore.current_step) {
        deepstackStore.highlight_action = actionIndex
    }
}

function actionMouseLeave(obsIndex: number, actionIndex: number) {
    if (actionIndex == deepstackStore.highlight_action) {
        deepstackStore.highlight_action = -1
    }
}

async function perform_action(obsIndex: number, actionIndex: number) {
    await deepstackStore.perform_action(obsIndex, actionIndex)
}

async function switch_obs(obsIndex: number) {
    deepstackStore.switch_obs(obsIndex)
}

function format_legal_actions(observation) {
    let result = []
    let actions = observation.legal_actions
    let action_name = [[0, "Fold"], [1, "Check"], [1, "Call"], [3, "All in"], [2, "Raise 100%"]]
    for (var i = 0; i < actions.length; i++) {
        if (actions[i] != null) {
            result.push(action_name[i])
        }
    }
    return result
}

</script>

<style scope>
.game-container {
    margin-bottom: 10px;
    color: white;
    font-weight: bold;
}

.observation-row {
    width: 100%;
    height: 130px;
    margin-bottom: 20px;
}

.action-container {
    background: rgb(30, 30, 30);
    width: 100px;
    height: 130px;
    border-radius: 4px;
    margin-right: 3px;
    padding: 2px 2px 0px 2px;
    float: left;
    cursor: pointer;
}

.action-container-s {
    background: rgb(50, 50, 50);
    border: 1px solid rgb(77, 169, 142);
}

.action-container .action-container-info {
    width: 100%;
    height: 22px;
    padding-left: 4px;
}

.action-container .action-player {
    color: rgb(170, 170, 170);
    font-weight: bolder;
    float: left;
    font-size: 12px;
}

.action-container-s .action-player {
    color: rgb(77, 169, 142);
}

.action-container .action-player-chips {
    color: rgb(170, 170, 170);
    padding-right: 4px;
    font-weight: bolder;
    float: right;
    font-size: 12px;
}

.action-container-s .action-player-chips {
    color: rgb(77, 169, 142);
}

.action-container .action-block-container {
    width: 100%;
    height: 100px;
    overflow-y: scroll;
}

.action-container .action-block {
    cursor: pointer;
    height: 17px;
    line-height: 17px;
    font-size: 14px;
    border-radius: 5px;
    padding-left: 4px;
    font-weight: bold;
}

.action-container-s .action-block {
    color: rgb(230, 230, 230)
}

.action-block:hover {
    background: rgb(100, 100, 100);
}

.action-block-s {
    background: rgb(100, 100, 100);
}

.board-card-row {
    width: 100%;
    height: 70px;
    margin-bottom: 10px;
}

.board-card-container {
    width: 40px;
    height: 60px;
    background: rgb(100, 100, 100);
    float: left;
    margin-right: 10px;
    text-align: center;
    border-radius: 6px;
}

.board-card-s {
    background : rgb(0, 0, 0);
}

.board-card-h {
    background : rgb(114, 29, 24);
}

.board-card-d {
    background : rgb(35, 72, 134);
}

.board-card-c {
    background : rgb(53, 117, 40);
}

.board-card {
    color: rgb(200, 200, 200);
    width: 100%;
    height: 100%;
    font-size: 25px;
    font-weight: bold;
    padding-top: 9px;
    border: 2px solid rgb(169, 169, 169);
    border-radius: 6px;
}


</style>