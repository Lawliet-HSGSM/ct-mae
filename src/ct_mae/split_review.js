"use strict";
const data=JSON.parse(document.getElementById('split-data').textContent);
const get=id=>document.getElementById(id);
for(const [index,title] of data.labels.entries())get('organ').add(new Option(title,index));
for(const [key,title] of Object.entries(data.metrics))get('metric').add(new Option(title,key));
get('organ').value='4';get('metric').value='dark';
get('recall-gap').src=data.figures.recall_gap;get('class-mix').src=data.figures.class_mix;
function card(key,title,augment){
 const article=document.createElement('article'),h=document.createElement('h3');
 h.textContent=title;article.append(h);
 if(!key){const p=document.createElement('p');p.className='empty';p.textContent='Không có mẫu ở vị trí này.';article.append(p);return article;}
 const c=data.cases[key];article.dataset.key=key;article.dataset.augmented=String(augment);
 const frame=document.createElement('div');frame.className='image-frame';
 const image=document.createElement('img'),size=Number(get('scale').value);
 image.src=augment?c.augmented[Number(get('view').value)]:c.image;
 image.width=size;image.height=size;image.style.width=size+'px';image.style.height=size+'px';
 image.alt=`${c.split} #${c.index}, ${data.labels[c.label]}${augment?', sau affine':''}`;
 frame.append(image);article.append(frame);
 const label=document.createElement('strong');label.textContent=`${c.split} #${c.index} · ${data.labels[c.label]}`;article.append(label);
 const line=document.createElement('div');
 if(c.models){const m=c.models[get('model').value],correct=m.prediction===c.label;
  line.className=correct?'good':'bad';line.textContent=`${correct?'Đúng':'Sai'} · ${data.labels[m.prediction]} · ${(100*m.confidence).toFixed(1)}%`;
 }else{line.className='meta';line.textContent=c.labeled5?'Thuộc 5% nhãn đã dùng':'Ngoài 5% nhãn gốc; ảnh từng dùng cho MAE';}
 article.append(line);return article;
}
function errorTable(){
 const organ=get('organ').value,metric=get('metric').value,model=get('model').value;
 const rows=data.conditional_errors[organ][metric],edges=rows[0].boundaries;
 get('bin-explanation').textContent=`${data.metrics[metric]} · mốc train: ${edges.map(x=>x.toFixed(4)).join(' / ')}. Tỷ lệ lỗi của ${model}; mọi ảnh val/test trong organ, không chỉ mẫu đang xem.`;
 const wrap=document.createElement('div');wrap.className='table-wrap';const table=document.createElement('table');
 const thead=document.createElement('thead');thead.innerHTML='<tr><th>Nhóm theo train</th><th>Số ảnh val</th><th>Val lỗi</th><th>Số ảnh test</th><th>Test lỗi</th></tr>';table.append(thead);
 const body=document.createElement('tbody');
 for(const [i,r] of rows.entries()){
  const tr=document.createElement('tr');
  const range=i===0?`x < ${edges[0].toFixed(4)}`:i===3?`x ≥ ${edges[2].toFixed(4)}`:`${edges[i-1].toFixed(4)} ≤ x < ${edges[i].toFixed(4)}`;
  const cells=[range,r.val.samples,r.val.models[model]===null?'—':(100*r.val.models[model]).toFixed(1)+'%',r.test.samples,r.test.models[model]===null?'—':(100*r.test.models[model]).toFixed(1)+'%'];
  for(const value of cells){const td=document.createElement('td');td.textContent=value;tr.append(td);}body.append(tr);
 }table.append(body);wrap.append(table);get('conditional-table').replaceChildren(wrap);
}
function render(){
 const organ=get('organ').value,pool=get('train-pool').value,model=get('model').value,mode=get('mode').value;
 const train=data.banks[pool][organ],val=data.banks.val[model][organ][mode],test=data.banks.test[model][organ][mode];
 const counts=[data.pool_counts[pool][organ],data.pool_counts.val[model][organ][mode],data.pool_counts.test[model][organ][mode]];
 get('pool-info').textContent=`${data.labels[organ]} · ứng viên train/val/test: ${counts.join(' / ')}. Mỗi bank tối đa16 mẫu, trang đang hiển thị tối đa8; mỗi organ có seed lấy mẫu cố định.`;
 const container=get('comparison');container.replaceChildren();const start=Number(get('page').value)*8;
 for(let i=start;i<start+8;i++){
  container.append(card(train[i],'Train gốc',false),card(train[i],'Cùng ảnh · augment',true),card(val[i],'Validation',false),card(test[i],'Test',false));
 }
 get('pixel-chart').src=data.figures['pixels_'+organ];errorTable();
}
for(const id of ['organ','train-pool','model','mode'])get(id).addEventListener('change',()=>{get('page').value='0';render();});
for(const id of ['page','view','scale'])get(id).addEventListener('change',render);
get('metric').addEventListener('change',errorTable);render();
