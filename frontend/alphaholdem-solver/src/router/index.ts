import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/deepstack',
      name: 'deepstack',
      component: () => import('../views/DeepStackView.vue')
    },
    {
      path: '/',
      name: 'study',
      component: () => import('../views/StudyView.vue')
    },
    {
      path: '/leduc',
      name: 'leduc',
      component: () => import('../views/LeducView.vue')
    },
  ]
})

export default router
