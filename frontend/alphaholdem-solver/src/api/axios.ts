import axios, { AxiosHeaders } from 'axios';
import { showMessage } from "./status"
import { ElMessage } from 'element-plus'
import { getEffectiveTypeParameterDeclarations } from 'typescript';

axios.defaults.timeout = 60000;

axios.defaults.baseURL = 'http://103.170.5.183:18889'

axios.interceptors.response.use(
  response => {
    return response;
  },
  error => {
    const { response } = error
    if (response) {
      showMessage(response.status)
      return Promise.reject(response.data)
    } else {
      ElMessage.warning('网络连接异常,请稍后再试!')
    }
  }
);

export function request(url = '', params = {}, method = 'POST', content_type='') {
  return new Promise((resolve, reject) => {
    axios({
      url: url,
      method: method.toUpperCase(),
      headers: {
        'Content-Type': content_type == 'json' ? 'application/json;charset=UTF-8': 'multipart/form-data'
      },
      params: params,
      data: params,
    }).then(res => {
      resolve(res)
    }).catch(err => {
      reject(err)
    })
  })
}