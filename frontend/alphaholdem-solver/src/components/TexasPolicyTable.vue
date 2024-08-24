<template>
    <div class="policy-table">
        <div class="overall-policy-table">
            <div v-for="(row, rowIndex) in overall_policy" :key="rowIndex" class="policy-row">
                <div v-for="(element, colIndex) in row" :key="colIndex" class="policy-element" @click="gameStore.switch_detail_cell(rowIndex, colIndex)">
                    <div class="policy-cell-mask"> </div>
                    <div class="policy-combo-name">
                        <span>{{ overall_cell_name[rowIndex][colIndex] }}</span>
                    </div>
                    <cus v-for="(prob, probIndex) in element.slice().reverse()" :key="probIndex">
                        <div :style="{width: prob * 100 + '%'}" class="policy-prob"
                            :class="(highlight_action >= 0 && element.length - probIndex - 1 != highlight_action ? 'policy-hide' : '') +
                                    ' policy-prob-' + (element.length - probIndex - 1)">
                        </div>
                    </cus>
                </div>
            </div>
        </div>
        <div class="detail-policy-table">
            <div v-for="(row, rowIndex) in detail_policy" :key="rowIndex" class="policy-row">
                <div v-for="(element, colIndex) in row" :key="colIndex" class="policy-element">
                    <div class="policy-combo-name">
                        <span>{{ detail_cell_name[rowIndex][colIndex] }}</span>
                    </div>
                    <cus v-for="(prob, probIndex) in element.slice().reverse()" :key="probIndex">
                        <div :style="{width: prob * 100 + '%'}" class="policy-prob"
                            :class="(highlight_action >= 0 && element.length - probIndex - 1 != highlight_action ? 'policy-hide' : '') +
                                    ' policy-prob-' + (element.length - probIndex - 1)">
                        </div>
                    </cus>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">

import { useTexasStore } from '@/stores/texas'
import { storeToRefs } from 'pinia'
const texasStore = useTexasStore()
texasStore.reset()

const {
    action_history,
    policy_history,
    observation_history,
    board_cards,
    overall_policy,
    overall_cell_name,
    highlight_action,
    detail_policy,
    detail_cell_name,
    policy_prior
} = storeToRefs(texasStore)

</script>

<style scope>
.policy-title {
    font-size: 25px;
    margin-bottom: 7px;
}

.policy-title span {
    font-weight: bold;
}

.policy-row {
    width: 770px;
    height: 40px;
    margin-bottom: 2px;
}

.policy-element {
    width: 55px;
    height: 40px;
    background:rgb(30, 30, 30);
    margin-left: 2px;
    margin-top: 2px;
    color: rgb(255, 255, 255);
    float: left;
    cursor: pointer;
}

.policy-combo-name {
    padding-left: 4px;
    width: 0;
    height: 0;
}


.policy-prob {
    height: 100%;
    float: left;
    background-color: red;
    width: 12.5%;
}

/* Fold */
.policy-prob-0 {
    background-color: rgb(77, 123, 179);
}

/* # Fold Check Call All_in Raise_25% Raise_50% Raise_75% Raise_125% */
/* Check */
.policy-prob-1 {
    background-color: rgb(114, 183, 111);
}

/* Call */
.policy-prob-2 {
    background-color: rgb(114, 183, 111);
}

/* All in */
.policy-prob-3 {
    background-color: rgb(48, 16, 14);
}

/* Raise 25% */
.policy-prob-4 {
    background-color: rgb(220, 76, 69);
}

/* Raise 50% */
.policy-prob-5 {
    background-color: rgb(186, 63, 57);
}

/* Raise 75% */
.policy-prob-6 {
    background-color: rgb(132, 45, 41);
}

/* Raise 125% */
.policy-prob-7 {
    background-color: rgb(115, 39, 35);
}

.policy-hide {
    opacity: 0.3;
}

/* .policy-cell-mask {
    width: 100%;
    height: 100%;
    background: red;
} */

.policy-table {
    width: 100%;
}

.overall-policy-table {
    float: left;
}

.detail-policy-table {
    width: 500px;
    height: 500px;
    float: left;
}

</style>