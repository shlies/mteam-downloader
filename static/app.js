const { createApp } = Vue

function fmtTime(s){
  if(!s) return '-'
  const d = new Date(s)
  return d.toLocaleString()
}

async function fetchJSON(url, options={}){
  const r = await fetch(url, options)
  if(!r.ok){
    const t = await r.text()
    throw new Error(t || r.status)
  }
  return r.json()
}

createApp({
  data(){
    return {
      activeTab: 'home',
      // state
      state: { running: false, last_poll_at: null, last_poll_message: '' },
      // saved searches
      saved: [],
  // preview
  new_keyword: '',
  new_task_name: '',
      preview_items: [],
  // config
  api_key: '', qb_url: '', qb_username: '', qb_password: '', download_base: '', poll_interval_sec: 300, enabled: true,
      qb_check_result: '',
      // downloads
      tasks: [],
      // account
      cp_current: '', cp_new: '', cp_confirm: '', cp_msg: '',
      // toasts
      toasts: [],
      toastId: 1,
    }
  },
  computed:{
    stateText(){
      return `运行中: ${this.state.running ? '是' : '否'} | 上次轮询: ${fmtTime(this.state.last_poll_at)} | 备注: ${this.state.last_poll_message || ''}`
    }
  },
  methods:{
    fmtTime,
    toast(message, type='info'){
      const id = this.toastId++
      this.toasts.push({id, message, type})
      setTimeout(()=>{
        const idx = this.toasts.findIndex(t=> t.id===id)
        if(idx>=0) this.toasts.splice(idx,1)
      }, 3000)
    },
    async loadState(){
      try{ this.state = await fetchJSON('/api/state') }catch(e){ /* ignore */ }
    },
    async loadConfig(){
      try{
        const c = await fetchJSON('/api/config')
        this.api_key = c.api_key || ''
        this.qb_url = c.qb_url || ''
        this.qb_username = c.qb_username || ''
        this.qb_password = ''
  this.download_base = c.download_base || ''
        this.poll_interval_sec = c.poll_interval_sec || 300
        this.enabled = !!c.enabled
      }catch(e){ /* ignore */ }
    },
    async saveConfig(){
      const body = {
        api_key: this.api_key === '***' ? undefined : (this.api_key || undefined),
        qb_url: this.qb_url || undefined,
        qb_username: this.qb_username || undefined,
        qb_password: this.qb_password || '',
        download_base: this.download_base || null,
        poll_interval_sec: this.poll_interval_sec || undefined,
        enabled: this.enabled,
      }
      Object.keys(body).forEach(k=> body[k] === undefined && delete body[k])
      try{
        await fetchJSON('/api/config', {method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)})
        this.toast('已保存', 'success')
        await this.loadState(); await this.loadTasks()
      }catch(e){ this.toast('保存失败：' + e.message, 'error') }
    },
    async trigger(){
      try{ await fetchJSON('/api/trigger', {method:'POST'}); this.toast('已触发一次轮询', 'success') }catch(e){ this.toast('触发失败：' + e.message, 'error') }
      await this.loadState(); await this.loadTasks()
    },
    async loadTasks(){
      try{ this.tasks = await fetchJSON('/api/downloads') }catch(e){ /* ignore */ }
    },
    async loadSaved(){
      try{ this.saved = await fetchJSON('/api/saved_searches') }catch(e){ /* ignore */ }
    },
    async preview(){
      if(!this.new_keyword){ this.toast('请输入关键词', 'warn'); return }
      try{
        const r = await fetchJSON('/api/preview', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({keyword: this.new_keyword})})
        this.preview_items = r.items || []
        if(!this.preview_items.length) this.toast('没有搜索结果', 'info')
      }catch(e){ this.toast('预览失败：' + e.message, 'error') }
    },
    async createSaved(){
      if(!this.new_keyword){ this.toast('请输入关键词', 'warn'); return }
      try{
        await fetchJSON('/api/saved_searches', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({keyword: this.new_keyword, task_name: this.new_task_name || null, enabled:true})})
        this.toast('已创建', 'success')
        this.new_keyword=''; this.new_task_name=''
        await this.loadSaved()
      }catch(e){ this.toast('创建失败：' + e.message, 'error') }
    },
    async toggleSaved(s){
      try{ await fetchJSON(`/api/saved_searches/${s.id}`, {method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify({enabled: s.enabled})}) }
      catch(e){ this.toast('更新失败：' + e.message, 'error') }
    },
    async delSaved(s){
      try{ await fetchJSON(`/api/saved_searches/${s.id}`, {method:'DELETE'}) ; this.toast('已删除', 'success'); await this.loadSaved() }
      catch(e){ this.toast('删除失败：' + e.message, 'error') }
    },
    async runSaved(s){
      try{ const r = await fetchJSON(`/api/saved_searches/${s.id}/run`, {method:'POST'}); this.toast(r.message || '已执行', 'success'); await this.loadTasks() }
      catch(e){ this.toast('执行失败：' + e.message, 'error') }
    },
    async qbCheck(){
      this.qb_check_result = '检查中...'
      try{ const r = await fetchJSON('/api/qb/check'); this.qb_check_result = `连接正常，版本：${r.version || 'unknown'}`; this.toast('qB 连接正常', 'success') }
      catch(e){ this.qb_check_result = `连接失败：${e.message}`; this.toast('qB 连接失败：' + e.message, 'error') }
    },
    async changePassword(){
      try{
        await fetchJSON('/api/change_password', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({current_password: this.cp_current, new_password: this.cp_new, confirm: this.cp_confirm})})
        this.cp_msg = '已修改密码'; this.toast('已修改密码', 'success')
        this.cp_current=this.cp_new=this.cp_confirm=''
      }catch(e){ this.cp_msg = e.message || '修改失败'; this.toast('修改失败：' + e.message, 'error') }
    },
    async logout(){
      try{ await fetchJSON('/api/logout', {method:'POST'}) }catch(e){ /* ignore */ }
      location.href = '/login'
    },
    async delTask(i){
      try{ await fetchJSON(`/api/downloads/${i.id}`, {method:'DELETE'}) ; this.toast('已删除记录', 'success'); await this.loadTasks() }
      catch(e){ this.toast('删除失败：' + e.message, 'error') }
    }
  },
  async mounted(){
    await this.loadConfig(); await this.loadState(); await this.loadTasks(); await this.loadSaved()
    setInterval(()=>{ this.loadState(); this.loadTasks() }, 15000)
  }
}).mount('#app')
// --- end ---
