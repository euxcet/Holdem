import './assets/main.css'

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { Button, Cell, CellGroup, Field, Uploader, Notify, Form, Popup, Icon } from 'vant'
import 'vant/lib/index.css'

import App from './App.vue'
import router from './router'

const app = createApp(App)

app.use(createPinia())
app.use(Button)
app.use(Cell)
app.use(CellGroup)
app.use(Field)
app.use(Uploader)
app.use(Notify)
app.use(Form)
app.use(Popup)
app.use(Icon)
app.use(router)

app.mount('#app')
