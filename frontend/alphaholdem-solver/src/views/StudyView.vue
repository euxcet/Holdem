<template>
    <main>
        <TexasActionHistory />
        <TexasPolicyTable :name='name' />
    </main>
</template>

<script setup lang="ts">
import TexasActionHistory from '../components/TexasActionHistory.vue'
import TexasPolicyTable from '../components/TexasPolicyTable.vue'
import { useRoute } from 'vue-router';

const name = useRoute().params.name
</script>