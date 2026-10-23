"use strict";
const review=JSON.parse(document.getElementById('review-data').textContent);
const get=id=>document.getElementById(id);
const names={scratch:'Scratch',linear:'Linear',finetune:'Full',partial:'Partial'};
for(const reference of review.references){
 const option=document.createElement('option');option.value=reference.index;
 option.textContent=`#${reference.index} · ${review.labels_vi[review.organ]}`;
 get('reference').append(option);
}
get('evidence').textContent=`Đã kiểm tra ${review.provenance.original_pngs_equal.length} ảnh tham chiếu trong báo cáo cũ: PNG trùng từng pixel với bản 128×128 chính thức. Trang này đã kiểm tra ${review.provenance.roundtrip_pngs_verified} PNG ở ba resolution; mọi ảnh giữ thang xám 0–255.`;
get('pool-counts').textContent=`Cùng nhãn ${review.labels_vi[review.organ]} có ${review.test_organ_count.toLocaleString('vi-VN')} ảnh test. Tập ứng viên: `+Object.entries(review.candidate_counts).map(([key,n])=>`${review.groups[key]}: ${n.toLocaleString('vi-VN')}`).join(' · ')+'.';
function imageFrame(caseData,size,display){
 const frame=document.createElement('div');frame.className='image-frame';
 const image=document.createElement('img');image.src=caseData.images[size];
 image.alt=`Test #${caseData.index}, nhãn ${review.labels_vi[caseData.label]}, ${size}×${size}`;
 image.width=display;image.height=display;image.style.width=display+'px';image.style.height=display+'px';
 image.style.imageRendering=get('smoothing').value;frame.append(image);return frame;
}
function card(index,title,display){
 const article=document.createElement('article'),heading=document.createElement('h3');
 heading.textContent=title;article.append(heading);
 if(index===null){const empty=document.createElement('p');empty.className='empty';
 empty.textContent='Không có ứng viên ở hạng này trong nhóm.';article.append(empty);return article;}
 const c=review.cases[index];article.dataset.index=index;
 article.append(imageFrame(c,'128',display));
 const strong=document.createElement('strong');strong.className='case-index';
 strong.textContent=`#${index} · Thật: ${review.labels_vi[c.label]}`;article.append(strong);
 for(const [stage,m] of Object.entries(c.models)){
  const line=document.createElement('div'),correct=m.prediction===c.label;
  line.className='prediction '+(correct?'good':'bad');
  line.textContent=`${names[stage]}: ${review.labels_vi[m.prediction]} · ${(m.confidence*100).toFixed(1)}% · ${correct?'đúng':'sai'}`;
  article.append(line);
 }
 return article;
}
function render(){
 const index=Number(get('reference').value),rank=Number(get('rank').value),scale=Number(get('scale').value);
 const reference=review.references.find(row=>row.index===index),display=128*scale;
 const container=get('comparisons');container.replaceChildren();
 container.style.setProperty('--card-min',Math.max(250,display+50)+'px');
 container.append(card(index,'Ảnh tham chiếu trong nhóm lỗi',display));
 for(const [key,title] of Object.entries(review.groups)){
  const match=reference.matches[key][rank];container.append(card(match?match.index:null,title,display));
 }
 get('selection').textContent=`Tham chiếu #${index} · hạng đối chiếu ${rank+1} · mỗi ảnh 128×128 đang hiển thị ${display}×${display} CSS px. Thay đổi cách xem không đổi input đã dùng để đánh giá model.`;
 get('resolutions').replaceChildren();
 for(const size of ['28','64','128']){
  const article=document.createElement('article'),heading=document.createElement('h3');
  heading.textContent=`Nguồn ${size}×${size} · cùng #${index}`;article.append(heading);
  article.append(imageFrame(review.cases[index],size,128));get('resolutions').append(article);
 }
}
for(const id of ['reference','rank','scale','smoothing'])get(id).addEventListener('change',render);
render();
