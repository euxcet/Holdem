import { defineStore } from 'pinia'
import { SolverService } from '@/api/api'
import { showNotify } from 'vant'

export const useTexasStore = defineStore('texas', {
    state: () => {
        return {
            // when need to select an action at step s, starting from 0.
            action_history: [] as number[], // s * 1
            policy_history: [], // (s + 1) * 1326 * num_actions
            policy_prior: [] as number[][], // s * 1326, prior after this step
            observation_history: [], // (s + 1) * ...
            board_cards: [] as string[],
            player_name: [] as string[],
            current_step: 0 as number,
            overall_policy: [] as number[][][],
            detail_policy: [] as number[][][],
            detail_cell_name: [] as string[][],
            overall_cell_name: [] as string[][],
            highlight_action: -1,
            running: false,
            num_actions: 0,
        }
    },
    actions: {
        empty_overall_policy() {
            return Array.from({ length: 13 }, () =>
                Array.from({ length: 13 }, () => Array(this.num_actions).fill(0)))
        },
        empty_detail_policy() {
            return Array.from({ length: 3 }, () =>
                Array.from({ length: 4 }, () => Array(this.num_actions).fill(0)))
        },

        get_overall_cell_name(): string[][] {
            let card_rank: string[] = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
            let cell_name = Array.from({ length: 13 }, () => Array(13).fill(""))
            for (var i = 0; i < 13 ; i++) {
                for (var j = 0; j < 13; j++) {
                    if (i < j) {
                        cell_name[i][j] = card_rank[i] + card_rank[j] + 's'
                    } else if (i == j) {
                        cell_name[i][j] = card_rank[i] + card_rank[j]
                    } else {
                        cell_name[i][j] = card_rank[j] + card_rank[i] + 'o'
                    }
                }
            }
            return cell_name
        },

        switch_detail_cell(row_index: number, col_index: number) {
            this.detail_policy = this.calc_detail_policy(
                this.policy_history[this.current_step],
                this.get_last_policy_prior(this.current_step),
                row_index,
                col_index,
            )
        },

        switch_obs(step: number) {
            if (this.running) {
                return
            }
            step = Math.min(step, this.action_history.length)
            this.current_step = step
            this.overall_policy = this.calc_overall_policy(
                this.policy_history[this.current_step], 
                this.get_last_policy_prior(this.current_step),
            )
        },

        async get_policy(): boolean {
            let response = (await SolverService.getPolicy({
                "action_history": this.action_history,
                "board_cards": this.board_cards,
            })).data
            return response
        },

        async reset() {
            // policy_history.length == observation_history.length == action_history.length + 1 == max_step + 1
            this.overall_cell_name = this.get_overall_cell_name()
            this.overall_policy = this.empty_overall_policy()
            this.detail_cell_name = Array.from({ length: 3 }, () => Array(4).fill(""))
            this.detail_policy = this.empty_detail_policy()
            this.action_history = []
            this.policy_history = []
            this.board_cards = ['Qh', '3h', 'As', '3s', 'Qs']
            let response = await this.get_policy()
            this.num_actions = response.policy[0].length
            this.policy_history.push(response.policy)
            this.observation_history.push(response.observation)
            this.current_step = 0
            this.overall_policy = this.calc_overall_policy(
                this.policy_history[this.current_step],
                this.get_last_policy_prior(this.current_step),
            )
            this.running = false
        },

        get_last_policy_prior(step: number): number[] {
            for (var s = step - 1; s >= 0; s--) {
                if (this.observation_history[s].current_player == this.observation_history[step].current_player) {
                    return this.policy_prior[s]
                }
            }
            return Array(1326).fill(1.0)
        },

        smooth_policy(policy: number[][]) {
            for (var i = 0; i < policy.length; i++) {
                var max_action_prob = 0
                var max_action_id = 0
                var sum = 0
                for (var j = 0; j < policy[i].length; j++) {
                    sum += policy[i][j]
                    if (policy[i][j] > max_action_prob) {
                        max_action_prob = policy[i][j]
                        max_action_id = j
                    }
                }
                for (var j = 0; j < policy[i].length; j++) {
                    if (policy[i][j] < sum * 0.05) {
                        policy[i][max_action_id] += policy[i][j]
                        policy[i][j] = 0
                    }
                }
            }
            return policy
        },

        async perform_action(step: number, action: number) {
            if (this.running) {
                return
            }
            this.running = true
            this.action_history = this.action_history.slice(0, step)
            this.policy_prior = this.policy_prior.slice(0, step)
            this.observation_history = this.observation_history.slice(0, step + 1)
            this.policy_history = this.policy_history.slice(0, step + 1)
            this.action_history.push(action)
            let prior = this.get_last_policy_prior(step)
            let new_prior = []
            for (var i = 0; i < prior.length; i++) {
                new_prior.push(prior[i] * this.policy_history[step][i][action])
            }
            this.policy_prior.push(new_prior)
            let response = await this.get_policy()

            if (!response.observation.is_over) {
                this.policy_history.push(this.smooth_policy(response.policy))
                this.observation_history.push(response.observation)
                this.current_step = step + 1
                let current_prior = this.get_last_policy_prior(this.current_step)

                // Remove combos that conflict with the board
                for (let card of response.observation.board_cards) {
                    var count = 0
                    for (var card0 = 0; card0 < 52; card0++) {
                        for (var card1 = card0 + 1; card1 < 52; card1++) {
                            if (card.suit_first_id == card0 || card.suit_first_id == card1) {
                                current_prior[count] = 0
                            }
                            count += 1
                        }
                    }
                }

                this.overall_policy = this.calc_overall_policy(
                    this.policy_history[this.current_step],
                    current_prior,)
            } else {
                showNotify('Game over!')
            }
            this.running = false
        },

        card_to_str(rank: number, suit: number): string {
            return ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"][rank] + ["c", "d", "h", "s"][suit]
        },

        calc_detail_policy(policy_list: number[][], policy_prior: number[], row_index: number, col_index: number): number[][][] {
            for (var i = 0; i < 3; i ++) {
                for (var j = 0; j < 4; j++) {
                    this.detail_cell_name[i][j] = ""
                }
            }
            let policy = this.empty_detail_policy()
            let current = 0
            let count = 0
            for (var card0 = 0; card0 < 52; card0++) {
                for (var card1 = card0 + 1; card1 < 52; card1++) {
                    let suit0 = Math.floor(card0 / 13)
                    let rank0 = card0 % 13
                    let suit1 = Math.floor(card1 / 13)
                    let rank1 = card1 % 13
                    if (suit0 == suit1) { // suited, div 4
                        if (rank0 < rank1) {
                            [rank0, rank1] = [rank1, rank0];
                            [suit0, suit1] = [suit1, suit0]
                        }
                    } else { // offsuit, div 12
                        if (rank0 > rank1) {
                            [rank0, rank1] = [rank1, rank0];
                            [suit0, suit1] = [suit1, suit0]
                        }
                    }
                    if (12 - rank0 == row_index && 12 - rank1 == col_index) {
                        if (rank0 < rank1) {
                            [rank0, rank1] = [rank1, rank0];
                            [suit0, suit1] = [suit1, suit0]
                        }
                        for (var i = 0; i < this.num_actions; i++) {
                            policy[Math.floor(count / 4)][count % 4][i] += policy_list[current][i] * policy_prior[current]
                            this.detail_cell_name[Math.floor(count / 4)][count % 4] =
                                this.card_to_str(rank0, suit0) + this.card_to_str(rank1, suit1)
                        }
                        count += 1
                    }
                    current += 1
                }
            }
            for (var i = 0; i < 3; i++) {
                for (var j = 0; j < 4; j++) {
                    let s = 0
                    for (var k = 0; k < this.num_actions; k++) {
                        policy[i][j][k] = Math.max(Math.min(policy[i][j][k], 0.999 - s), 0)
                        s += policy[i][j][k]
                    }
                    if (s > 0.01) {
                        for (var k = 0; k < this.num_actions; k++) {
                            policy[i][j][k] = policy[i][j][k] / s
                        }
                    }
                }
            }
            return policy
        },

        // policy_list: 1326 * action_num   policy_prior: 1326
        calc_overall_policy(policy_list: number[][], policy_prior: number[]): number[][][] {
            let policy = this.empty_overall_policy()
            let current = 0
            for (var card0 = 0; card0 < 52; card0++) {
                for (var card1 = card0 + 1; card1 < 52; card1++) {
                    let suit0 = Math.floor(card0 / 13)
                    let rank0 = card0 % 13
                    let suit1 = Math.floor(card1 / 13)
                    let rank1 = card1 % 13
                    let ratio = 0.0
                    if (rank0 == rank1) { // pair, div 6
                        ratio = 1.0 / 6.0
                    } else if (suit0 == suit1) { // suited, div 4
                        ratio = 1.0 / 4.0
                        if (rank0 < rank1) {
                            [rank0, rank1] = [rank1, rank0];
                            [suit0, suit1] = [suit1, suit0]
                        }
                    } else { // offsuit, div 12
                        ratio = 1.0 / 12.0
                        if (rank0 > rank1) {
                            [rank0, rank1] = [rank1, rank0];
                            [suit0, suit1] = [suit1, suit0]
                        }
                    }
                    for (var i = 0; i < this.num_actions; i++) {
                        policy[12 - rank0][12 - rank1][i] += policy_list[current][i] * ratio * policy_prior[current]
                    }
                    current += 1
                }
            }
            for (var i = 0; i < 13; i++) {
                for (var j = 0; j < 13; j++) {
                    let s = 0
                    for (var k = 0; k < this.num_actions; k++) {
                        policy[i][j][k] = Math.max(Math.min(policy[i][j][k], 0.999 - s), 0)
                        s += policy[i][j][k]
                    }
                    if (s > 0.01) {
                        for (var k = 0; k < this.num_actions; k++) {
                            policy[i][j][k] = policy[i][j][k] / s
                        }
                    }
                }
            }
            return policy
        }
    },
})