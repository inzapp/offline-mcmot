"""CSV summaries of snapshot predictions (not unique people or accuracy)."""
import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev


def age_band(age):
    if age < 0:
        return 'invalid_negative'
    return f'{int(age // 10) * 10}-{int(age // 10) * 10 + 9}' if age < 80 else '80+'


def write_statistics(results, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    good = [r for r in results if r.get('status') == 'ok']

    def save(name, fields, rows):
        with (output / name).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def numeric(key):
        values = []
        for r in good:
            try:
                v = float(r[key])
                if math.isfinite(v):
                    values.append(v)
            except (KeyError, TypeError, ValueError):
                pass
        return values

    def distribution(name, counter, categories, denominator):
        save(name, ['category','count','percent','denominator'],
             [{'category':k,'count':counter[k],
               'percent':round(counter[k]*100/denominator,4) if denominator else 0,
               'denominator':denominator} for k in categories])

    summary = [{'metric':'total_snapshots','value':len(results)},
               {'metric':'successful_snapshots','value':len(good)},
               {'metric':'failed_snapshots','value':len(results)-len(good)}]
    for key in ['age','gender_score','detection_ms','preprocess_ms','inference_ms','pipeline_ms']:
        vals = sorted(numeric(key))
        metrics = {'count':len(vals),'mean':mean(vals) if vals else '',
                   'median':median(vals) if vals else '',
                   'min':min(vals) if vals else '', 'max':max(vals) if vals else '',
                   'std_population':pstdev(vals) if vals else ''}
        if key.endswith('_ms'):
            metrics['p95_nearest_rank'] = vals[math.ceil(.95*len(vals))-1] if vals else ''
        summary.extend({'metric':f'{key}_{k}','value':round(v,4) if isinstance(v,float) else v}
                       for k,v in metrics.items())
    save('summary.csv',['metric','value'],summary)
    bands = [f'{n}-{n+9}' for n in range(0,80,10)] + ['80+','invalid_negative']
    ages = numeric('age')
    distribution('age_distribution.csv', Counter(map(age_band,ages)), bands, len(ages))
    for field, filename, defaults in [
        ('gender','gender_distribution.csv',['male','female']),
        ('input_mode','input_mode_distribution.csv',['face_body','body_only']),
        ('face_status','face_status_distribution.csv',['detected','no_usable_face','multiple_faces','disabled'])]:
        counts = Counter(str(r.get(field,'unknown')) for r in good)
        distribution(filename,counts,defaults+sorted(set(counts)-set(defaults)),len(good))
    cross = Counter()
    for r in good:
        try:
            age = float(r['age'])
            if math.isfinite(age):
                cross[(age_band(age),str(r.get('gender','unknown')))] += 1
        except (KeyError,TypeError,ValueError):
            pass
    genders = sorted({'male','female'} | {k[1] for k in cross})
    save('age_gender_distribution.csv',['age_group','gender','count','percent_of_valid_age'],
         [{'age_group':b,'gender':g,'count':cross[b,g],
           'percent_of_valid_age':round(cross[b,g]*100/len(ages),4) if ages else 0}
          for b in bands for g in genders])
    save('errors.csv',['file','error'],[{'file':r.get('file',''), 'error':r.get('error','unknown')}
                                      for r in results if r.get('status') != 'ok'])
