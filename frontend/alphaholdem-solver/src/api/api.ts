import { request } from './axios'

export class SolverService {
  static async getPolicy(params) {
    return request('/policy', params, 'post', 'json')
  }
}