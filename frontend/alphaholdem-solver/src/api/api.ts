import { request } from './axios'

export class SolverService {
  static async getDeepStackNumGames(params) {
    return request('/deepstack_num_games', params, 'get', 'json')
  }

  static async getPolicy(params) {
    return request('/policy', params, 'post', 'json')
  }

  static async getFixPreflopPolicy(params) {
    return request('/fix_preflop_policy', params, 'post', 'json')
  }

  static async getStrengthPolicy(params) {
    return request('/strength_policy', params, 'post', 'json')
  }

  static async getRangePolicy(params) {
    return request('/range_policy', params, 'post', 'json')
  }

  static async getDeepStackPolicy(params) {
    return request('/deepstack_policy', params, 'post', 'json')
  }
}