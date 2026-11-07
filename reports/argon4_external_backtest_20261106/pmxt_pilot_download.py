"""Bounded public-only PMXT feasibility acquisition. Does not place orders or score PnL."""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import re
import struct
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "pmxt_pilot"
URL = "https://r2v2.pmxt.dev/polymarket_orderbook_2026-04-17T12.parquet"
CAP = 128 * 1024 * 1024
DATASET = "argon4_pmxt_external_replay_pilot_20261106"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Immutable artifact conflict: {path}")
        return
    with path.open("xb") as f:
        f.write(data)


def save_json(path, value):
    save_once(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


class Fetcher:
    def __init__(self, metadata):
        self.s = requests.Session()
        self.s.trust_env = False
        self.s.auth = None
        self.s.proxies.clear()
        self.s.headers.update({"Accept-Encoding": "identity"})
        self.etag = metadata["receipts"][0]["headers"]["ETag"]
        self.modified = metadata["receipts"][0]["headers"]["Last-Modified"]
        self.total = metadata["object_bytes"]
        self.used = 0
        self.receipts = []

    def fetch(self, url, path, *, params=None, start=None, end=None, maximum=2*1024*1024):
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.username or parts.password or parts.fragment:
            raise ValueError("Only credential-free HTTPS allowed")
        if not ((url == URL and start is not None and params is None) or
                (url == "https://gamma-api.polymarket.com/markets" and start is None) or
                (re.fullmatch(r'https://gamma-api\.polymarket\.com/markets/slug/(btc|eth)-updown-(5m|15m)-[0-9]+',url) and start is None and params is None)):
            raise ValueError("Endpoint not whitelisted")
        receipt_path = path.with_suffix(path.suffix + ".receipt.json")
        if path.exists() or receipt_path.exists():
            if not (path.exists() and receipt_path.exists()):
                raise ValueError("Incomplete immutable acquisition; manual review required")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            data = path.read_bytes()
            if sha(data) != receipt["sha256"] or receipt["request_url"] != url or receipt["params"] != params or receipt["range"] != ([start,end] if start is not None else None):
                raise ValueError("Cached acquisition identity mismatch")
            if start is not None and (receipt["headers"].get("ETag") != self.etag or len(data) != end-start+1):
                raise ValueError("Cached range version mismatch")
            self.used += len(data)
            if self.used > CAP:
                raise ValueError("Total source bytes exceeded")
            self.receipts.append(receipt)
            return data
        wanted = end-start+1 if start is not None else maximum
        if self.used + wanted > CAP:
            raise ValueError("Total source byte cap would be exceeded")
        headers = {"Accept-Encoding": "identity"}
        if start is not None:
            headers.update(Range=f"bytes={start}-{end}", **{"If-Match":self.etag})
        self.s.cookies.clear()
        at = utc()
        with self.s.get(url, params=params, headers=headers, timeout=(10,60), stream=True, allow_redirects=False) as response:
            receipt = {"request_url":url,"final_url":response.url,"params":params,"range":[start,end] if start is not None else None,"requested_at_utc":at,"headers_received_at_utc":utc(),"status":response.status_code,"headers":dict(response.headers)}
            if response.status_code not in ({206} if start is not None else {200}):
                receipt['body_read']=False
                save_json(OUT/'failures'/(dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),receipt)
                raise ValueError('HTTP response rejected without retry: '+str(response.status_code))
            if start is not None:
                if response.status_code != 206:
                    raise ValueError("Non-206 response rejected before reading body")
                if response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.total}" or response.headers.get("ETag") != self.etag or response.headers.get("Last-Modified") != self.modified:
                    raise ValueError("Range/version/length identity mismatch")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Encoded range rejected")
            elif response.status_code != 200 or "json" not in response.headers.get("Content-Type", ""):
                raise ValueError("Metadata response rejected")
            data = response.raw.read(wanted + 1)
            if (start is not None and len(data) != wanted) or (start is None and len(data) > maximum):
                raise ValueError("Bounded response length mismatch")
            self.used += len(data)
            receipt.update(received_at_utc=utc(),bytes=len(data),sha256=sha(data),local_path=str(path))
        save_once(path, data)
        save_json(receipt_path, receipt)
        self.receipts.append(receipt)
        return data


class Segments(io.RawIOBase):
    """Expose only real fetched offsets. Never fabricate unrequested parquet bytes."""
    def __init__(self, total, parts):
        super().__init__()
        self.total = total
        self.parts = sorted(parts)
        self.pos = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        p = offset if whence == 0 else self.pos+offset if whence == 1 else self.total+offset
        if p < 0 or p > self.total: raise ValueError("Seek outside object")
        self.pos = p
        return p
    def read(self, n=-1):
        if n < 0: n = self.total-self.pos
        end = min(self.total, self.pos+n)
        chunks=[]
        while self.pos < end:
            match = next(((a,b) for a,b in self.parts if a <= self.pos < a+len(b)), None)
            if match is None: raise ValueError(f"Attempted read of unfetched bytes at {self.pos}")
            a,b = match; take=min(end-self.pos,len(b)-(self.pos-a))
            chunks.append(b[self.pos-a:self.pos-a+take]); self.pos += take
        return b"".join(chunks)
    def readinto(self, b):
        d=self.read(len(b)); b[:len(d)]=d; return len(d)


def decode_list(value):
    result=json.loads(value) if isinstance(value,str) else value
    if not isinstance(result,list): raise ValueError("Metadata list missing")
    return result


def main():
    registry=json.loads((ROOT/'archive/catalog/dataset_registry.json').read_text(encoding='utf-8'))
    entry=next(x for x in registry['datasets'] if x.get('id')==DATASET)
    if Path(entry['source_path']).resolve()!=OUT.resolve() or entry['convertible'] is not False:
        raise ValueError('Dataset registration identity mismatch')
    signal_path=HERE/'pilot_frozen_signals.json'; signal_bytes=signal_path.read_bytes(); frozen=json.loads(signal_bytes)
    if frozen['interval_start_inclusive']!='2026-04-17T12:00:00+00:00' or frozen['interval_end_exclusive']!='2026-04-17T13:00:00+00:00' or len(frozen['selected_signals'])!=5:
        raise ValueError('Frozen experiment window mismatch')
    metadata=json.loads((HERE/'pmxt_footer_metadata.json').read_text(encoding='utf-8'))
    f=Fetcher(metadata); markets=[]
    try:
        for sig in frozen['selected_signals']:
            asset,interval=sig['stream'].split('_'); timestamp=int(dt.datetime.fromisoformat(sig['window_start_utc']).timestamp())
            slug=f'{asset.lower()}-updown-{interval}-{timestamp}'
            metadata_path=OUT/'gamma'/f'{slug}.slug.json'
            raw=f.fetch('https://gamma-api.polymarket.com/markets/slug/'+slug,metadata_path)
            market=json.loads(raw)
            if not isinstance(market,dict) or market.get('slug')!=slug:
                raise ValueError('Official slug response not unique/exact')
            outcomes=decode_list(market.get('outcomes'));tokens=decode_list(market.get('clobTokenIds'))
            if len(outcomes)!=2 or set(x.upper() for x in outcomes)!={'UP','DOWN'} or len(tokens)!=2 or len(set(tokens))!=2 or any(not str(x).isdigit() for x in tokens):
                raise ValueError('Invalid binary token mapping')
            cid=market.get('conditionId','')
            if not re.fullmatch('0x[0-9a-fA-F]{64}',cid):raise ValueError('Invalid condition identity')
            if dt.datetime.fromisoformat(market['endDate'].replace('Z','+00:00'))!=dt.datetime.fromisoformat(sig['window_end_utc']):
                raise ValueError('Official market end window mismatch')
            row={'signal':sig,'slug':slug,'gamma_id':str(market['id']),'condition_id':cid,'outcome_tokens':dict(zip((x.upper() for x in outcomes),map(str,tokens))),'gamma_raw_path':str(metadata_path),'gamma_raw_sha256':sha(raw),'gamma_metadata_received_at':f.receipts[-1]['received_at_utc'],'current_metadata_only_not_historical_receipt':True}
            row['metadata_summary']={k:market.get(k) for k in ['question','description','eventStartTime','startDate','endDate','closed','closedTime','umaResolutionStatus','resolutionSource','outcomes','outcomePrices','feesEnabled','feeSchedule','makerBaseFee','takerBaseFee','feeType','orderPriceMinTickSize','orderMinSize','updatedAt']}
            markets.append(row)
        save_json(OUT/'market_metadata.json',markets)
        tail=f.fetch(URL,OUT/'ranges'/f'{f.total-8}-{f.total-1}.bin',start=f.total-8,end=f.total-1)
        if tail[4:]!=b'PAR1':raise ValueError('Bad parquet trailer')
        n=struct.unpack('<I',tail[:4])[0]
        if n>2*1024*1024 or n!=metadata['footer_bytes']:raise ValueError('Unexpected footer')
        footer_start=f.total-8-n
        footer=f.fetch(URL,OUT/'ranges'/f'{footer_start}-{f.total-9}.bin',start=footer_start,end=f.total-9)
        if sha(footer)!=metadata['receipts'][1]['sha256']:raise ValueError('Footer content changed since audit')
        magic=f.fetch(URL,OUT/'ranges'/'0-3.bin',start=0,end=3)
        if magic!=b'PAR1':raise ValueError('Bad header')
        groups=[]; mappings={}
        for market in markets:
            eligible=[]
            for g in metadata['groups']:
                stats=next(c['statistics'] for c in g['columns'] if c['name']=='market')
                if stats['min']<=market['condition_id']<=stats['max']:eligible.append(g['index'])
            mappings[market['slug']]=eligible;groups.extend(eligible)
        groups=sorted(set(groups));parts=[(0,magic),(footer_start,footer+tail)];ranges=[]
        for i in groups:
            g=metadata['groups'][i]
            start=min(c['dictionary_page_offset'] if c['dictionary_page_offset'] is not None else c['data_page_offset'] for c in g['columns'])
            end=max((c['dictionary_page_offset'] if c['dictionary_page_offset'] is not None else c['data_page_offset'])+c['compressed_bytes']-1 for c in g['columns'])
            ranges.append({'row_group':i,'start':start,'end':end,'bytes':end-start+1})
        if sum(x['bytes'] for x in ranges)+f.used>CAP:raise ValueError('Planned groups exceed source byte budget')
        save_json(OUT/'range_plan.json',{'url':URL,'etag':f.etag,'total':f.total,'candidate_groups':mappings,'ranges':ranges,'cap':CAP,'frozen_signals_sha256':sha(signal_bytes)})
        for plan in ranges:
            data=f.fetch(URL,OUT/'ranges'/f"{plan['start']}-{plan['end']}.bin",start=plan['start'],end=plan['end'])
            parts.append((plan['start'],data));print(json.dumps({'downloaded_group':plan['row_group'],'source_bytes':f.used}),flush=True)
        pf=pq.ParquetFile(Segments(f.total,parts),pre_buffer=False)
        targets=pa.array([m['condition_id'].encode() for m in markets], type=pa.binary(66))
        extracts=[]
        for g in groups:
            table=pf.read_row_group(g,use_threads=False)
            index=pa.array(range(table.num_rows),type=pa.int64())
            table=table.append_column('source_row_group',pa.array([g]*table.num_rows,type=pa.int32())).append_column('source_row_in_group',index)
            selected=table.filter(pc.is_in(table['market'],value_set=targets))
            dst=OUT/'derived'/f'target_events_rg{g:03d}.parquet'
            sink=pa.BufferOutputStream();pq.write_table(selected,sink,compression='zstd');raw=sink.getvalue().to_pybytes();save_once(dst,raw)
            extracts.append({'row_group':g,'path':str(dst),'sha256':sha(raw),'rows':selected.num_rows,'candidate_rows':table.num_rows})
            print(json.dumps({'extracted_group':g,'target_rows':selected.num_rows}),flush=True)
        receipt={'status':'DOWNLOADED_AND_FILTERED_NO_PNL','dataset_id':DATASET,'frozen_signals_sha256':sha(signal_bytes),'script_sha256':sha(Path(__file__).read_bytes()),'completed_at_utc':utc(),'cap_source_bytes':CAP,'total_source_bytes':f.used,'url':URL,'object_bytes':f.total,'etag':f.etag,'last_modified':f.modified,'source_receipts':f.receipts,'extracted_files':extracts,'market_mapping':markets,'source_sequence_preserved':['source_row_group','source_row_in_group'],'profit_estimated':False,'production_access':False}
        save_json(OUT/'acquisition.json',receipt)
        print(json.dumps({'status':receipt['status'],'source_bytes':f.used,'target_rows':sum(x['rows'] for x in extracts),'groups':groups}),flush=True)
    finally:
        f.s.close()


if __name__=='__main__':
    main()
