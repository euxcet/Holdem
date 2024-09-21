import { request } from './axios'

export class SolverService {
  static async getPolicy(params) {
    return request('/policy', params, 'post', 'json')
  }

  static async getDeepStackPolicy(params) {
    return request('/deepstack_policy', params, 'post', 'json')
  }
}