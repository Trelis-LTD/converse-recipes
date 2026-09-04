import { DialtClient } from 'https://cdn.jsdelivr.net/npm/@dialt/sdk@0.26.0/src/index.js';

const plan = await fetch('./plan.json').then(response => response.json());
const answers = {};
const fields = document.querySelector('#fields');
const transcript = document.querySelector('#transcript');
const esc = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const instructions = `${plan.objective}\nCollect these fields naturally and call record_plan_field whenever one is supported or corrected. Do not read the list aloud and do not finish before every required field is recorded.\n${plan.fields.map(field => `- ${field.key} (${field.required === false ? 'optional' : 'required'}): ${field.description}`).join('\n')}\nWhen complete: ${plan.completion}`;
const tool = {name:'record_plan_field',description:'Record or correct supported interview evidence.',parameters:{type:'object',properties:{field:{type:'string',enum:plan.fields.map(field=>field.key)},value:{type:'string'}},required:['field','value']},expected_duration:'instant',status_label:'interview notes'};
let client;

function render() {
  fields.innerHTML = plan.fields.map(field => `<div class="py-2 border-bottom ${answers[field.key]?'done-field':''}"><strong>${esc(field.key.replaceAll('_',' '))}</strong><br><small>${esc(answers[field.key] || field.description)}</small></div>`).join('');
  document.querySelector('#answers').textContent = JSON.stringify(answers, null, 2);
}
function addTurn(role,text){transcript.insertAdjacentHTML('beforeend',`<div class="turn bg-light p-3 mb-2"><strong>${esc(role)}</strong><br>${esc(text)}</div>`)}
render();

document.querySelector('#start').onclick = async () => {
  const modality = document.querySelector('#modality').value;
  client = new DialtClient({url:document.querySelector('#url').value,sessionId:document.querySelector('#session').value,apiKey:document.querySelector('#key').value,mode:{kind:'dialt',modality,instructions,tools:[tool],greeting:'Tell me about your role and the last urgent customer escalation you handled.'}});
  client.addEventListener('asr', event => addTurn('you', event.detail.text));
  client.addEventListener('utterance', event => addTurn('assistant', event.detail.text));
  client.addEventListener('tool_call', event => {
    const {id,name,args} = event.detail;
    if(name !== 'record_plan_field') return;
    answers[args.field] = args.value; render();
    const missing = plan.fields.filter(field => field.required !== false && !answers[field.key]).map(field => field.key);
    client.sendToolResult(id,{recorded:args.field,missing_required:missing,complete:missing.length===0},{outcome:'succeeded',verified:true});
  });
  if(modality === 'voice') await client.unlockAudio();
  await client.connect();
  if(modality === 'voice') await client.startMic();
  document.querySelector('#composer').hidden = modality === 'voice';
  document.querySelector('#start').disabled = true;
};
document.querySelector('#composer').onsubmit = event => {event.preventDefault();const input=document.querySelector('#message');if(client&&input.value.trim()){client.sendText(input.value);input.value=''}};
