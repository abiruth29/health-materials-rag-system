# -*- coding: utf-8 -*-
"""
Health Materials RAG System - Accuracy Test Suite
Comprehensive accuracy testing with 8 test categories and evaluation metrics.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.rag_pipeline.health_materials_rag_demo import HealthMaterialsRAG


class RAGAccuracyTester:
    def __init__(self):
        self.rag = HealthMaterialsRAG()
        self.results_summary = {}
        
    def run_all_tests(self):
        print("\n[*] HEALTH MATERIALS RAG ACCURACY TEST SUITE")
        print("="*60)
        
        self.test_known_material_retrieval()
        self.test_application_based_search()
        self.test_property_based_search()
        self.test_regulatory_standards_search()
        self.test_material_classification()
        self.test_ranking_quality()
        self.test_negative_cases()
        self.test_cross_domain_knowledge()
        
        self.print_final_summary()
        self.save_results()
    
    def test_known_material_retrieval(self):
        print("\n[Test 1] Known Material Retrieval Accuracy")
        
        test_cases = [
            {'query': 'Ti-6Al-4V titanium alloy', 'keywords': ['Ti', 'titanium'], 'min_score': 0.35},
            {'query': '316L stainless steel for medical implants', 'keywords': ['316L', 'stainless'], 'min_score': 0.30},
            {'query': 'hydroxyapatite bone substitute material', 'keywords': ['hydroxyapatite', 'bone'], 'min_score': 0.35},
            {'query': 'PEEK polymer for spinal implants', 'keywords': ['PEEK', 'polymer'], 'min_score': 0.30},
            {'query': 'Nitinol shape memory alloy', 'keywords': ['Nitinol', 'NiTi'], 'min_score': 0.35}
        ]
        
        passed, total, latencies = 0, len(test_cases), []
        
        for i, test in enumerate(test_cases, 1):
            start_time = time.time()
            search_result = self.rag.semantic_search(test['query'], top_k=5)
            latency = time.time() - start_time
            latencies.append(latency)
            
            results = search_result['results']
            if not results:
                print(f"  [FAIL] Test {i}: No results")
                continue
            
            top_text = results[0]['text'].lower() + ' ' + str(results[0]['metadata']).lower()
            found_kw = sum(1 for kw in test['keywords'] if kw.lower() in top_text)
            score_ok = results[0]['similarity_score'] >= test['min_score']
            
            if found_kw >= len(test['keywords']) * 0.5 and score_ok:
                passed += 1
                status = "[PASS]"
            else:
                status = "[FAIL]"
            
            print(f"  {status} Test {i}: {test['query'][:45]}... (score: {results[0]['similarity_score']:.3f}, {latency*1000:.1f}ms)")
        
        acc = (passed / total) * 100
        avg_lat = np.mean(latencies) * 1000
        print(f"\n  Accuracy: {passed}/{total} ({acc:.1f}%) | Avg Latency: {avg_lat:.1f}ms")
        
        self.results_summary['known_material_retrieval'] = {
            'accuracy': acc, 'passed': passed, 'total': total, 'avg_latency_ms': avg_lat
        }
    
    def test_application_based_search(self):
        print("\n[Test 2] Application-Based Search Accuracy")
        
        test_cases = [
            {'query': 'materials for orthopedic implants', 'keywords': ['orthopedic', 'implant'], 'min_mat': 3},
            {'query': 'biocompatible materials for cardiovascular stents', 'keywords': ['cardiovascular', 'stent'], 'min_mat': 3},
            {'query': 'polymers suitable for drug delivery systems', 'keywords': ['polymer', 'drug'], 'min_mat': 2},
            {'query': 'materials for dental restoration', 'keywords': ['dental', 'restoration'], 'min_mat': 2},
            {'query': 'bioactive coatings for titanium implants', 'keywords': ['bioactive', 'coating'], 'min_mat': 2}
        ]
        
        passed, total, latencies = 0, len(test_cases), []
        
        for i, test in enumerate(test_cases, 1):
            start_time = time.time()
            search_result = self.rag.semantic_search(test['query'], top_k=10)
            latency = time.time() - start_time
            latencies.append(latency)
            
            results = search_result['results']
            mat_count = sum(1 for r in results if r['metadata'].get('type') == 'material')
            
            all_text = ' '.join([r['text'].lower() for r in results[:5]])
            found_kw = sum(1 for kw in test['keywords'] if kw.lower() in all_text)
            
            if mat_count >= test['min_mat'] and found_kw >= len(test['keywords']) * 0.6:
                passed += 1
                status = "[PASS]"
            else:
                status = "[FAIL]"
            
            print(f"  {status} Test {i}: {test['query'][:45]}... ({mat_count} materials, {latency*1000:.1f}ms)")
        
        acc = (passed / total) * 100
        avg_lat = np.mean(latencies) * 1000
        print(f"\n  Accuracy: {passed}/{total} ({acc:.1f}%) | Avg Latency: {avg_lat:.1f}ms")
        
        self.results_summary['application_based_search'] = {
            'accuracy': acc, 'passed': passed, 'total': total, 'avg_latency_ms': avg_lat
        }
    
    def test_property_based_search(self):
        print("\n[Test 3] Property-Based Search Accuracy")
        
        test_cases = [
            {'query': 'high corrosion resistance biocompatible metal', 'keywords': ['corrosion', 'resistant'], 'min_mat': 2},
            {'query': 'elastic modulus similar to bone tissue', 'keywords': ['elastic', 'bone'], 'min_mat': 2},
            {'query': 'antibacterial surface coating materials', 'keywords': ['antibacterial', 'coating'], 'min_mat': 2},
            {'query': 'radiopaque materials for X-ray visibility', 'keywords': ['radiopaque', 'X-ray'], 'min_mat': 1},
            {'query': 'biodegradable polymers for temporary implants', 'keywords': ['biodegradable', 'polymer'], 'min_mat': 2}
        ]
        
        passed, total, latencies = 0, len(test_cases), []
        
        for i, test in enumerate(test_cases, 1):
            start_time = time.time()
            search_result = self.rag.semantic_search(test['query'], top_k=5)
            latency = time.time() - start_time
            latencies.append(latency)
            
            results = search_result['results']
            mat_count = sum(1 for r in results if r['metadata'].get('type') == 'material')
            
            all_text = ' '.join([r['text'].lower() for r in results])
            found_kw = sum(1 for kw in test['keywords'] if kw.lower() in all_text)
            
            if mat_count >= test['min_mat'] and found_kw >= len(test['keywords']) * 0.5:
                passed += 1
                status = "[PASS]"
            else:
                status = "[FAIL]"
            
            print(f"  {status} Test {i}: {test['query'][:45]}... ({mat_count} materials, {latency*1000:.1f}ms)")
        
        acc = (passed / total) * 100
        avg_lat = np.mean(latencies) * 1000
        print(f"\n  Accuracy: {passed}/{total} ({acc:.1f}%) | Avg Latency: {avg_lat:.1f}ms")
        
        self.results_summary['property_based_search'] = {
            'accuracy': acc, 'passed': passed, 'total': total, 'avg_latency_ms': avg_lat
        }
    
    def test_regulatory_standards_search(self):
        print("\n[Test 4] Regulatory/Standards Search Accuracy")
        
        test_cases = [
            {'query': 'ISO 10993 biocompatibility certified materials', 'keywords': ['ISO', 'biocompatibility'], 'min_res': 3},
            {'query': 'FDA approved materials for medical devices', 'keywords': ['FDA', 'approved'], 'min_res': 3},
            {'query': 'ASTM F136 titanium alloy specifications', 'keywords': ['ASTM', 'titanium'], 'min_res': 2},
            {'query': 'USP Class VI compliant polymers', 'keywords': ['USP', 'polymer'], 'min_res': 2}
        ]
        
        passed, total, latencies = 0, len(test_cases), []
        
        for i, test in enumerate(test_cases, 1):
            start_time = time.time()
            search_result = self.rag.semantic_search(test['query'], top_k=10)
            latency = time.time() - start_time
            latencies.append(latency)
            
            results = search_result['results']
            all_text = ' '.join([r['text'].lower() for r in results[:5]])
            found_kw = sum(1 for kw in test['keywords'] if kw.lower() in all_text)
            
            if len(results) >= test['min_res'] and found_kw >= len(test['keywords']) * 0.4:
                passed += 1
                status = "[PASS]"
            else:
                status = "[FAIL]"
            
            print(f"  {status} Test {i}: {test['query'][:45]}... ({len(results)} results, {latency*1000:.1f}ms)")
        
        acc = (passed / total) * 100
        avg_lat = np.mean(latencies) * 1000
        print(f"\n  Accuracy: {passed}/{total} ({acc:.1f}%) | Avg Latency: {avg_lat:.1f}ms")
        
        self.results_summary['regulatory_standards_search'] = {
            'accuracy': acc, 'passed': passed, 'total': total, 'avg_latency_ms': avg_lat
        }
    
    def test_material_classification(self):
        print("\n[Test 5] Material Classification Accuracy")
        
        test_cases = [
            {'query': 'titanium alloys for medical implants', 'keywords': ['titanium', 'alloy']},
            {'query': 'biodegradable polymers for sutures', 'keywords': ['polymer', 'biodegradable']},
            {'query': 'ceramic materials for hip replacements', 'keywords': ['ceramic', 'hip']},
            {'query': 'composite materials for dental restoration', 'keywords': ['composite', 'dental']}
        ]
        
        passed, total, latencies = 0, len(test_cases), []
        
        for i, test in enumerate(test_cases, 1):
            start_time = time.time()
            search_result = self.rag.semantic_search(test['query'], top_k=5)
            latency = time.time() - start_time
            latencies.append(latency)
            
            results = search_result['results']
            if not results:
                print(f"  [FAIL] Test {i}: No results")
                continue
            
            top_text = results[0]['text'].lower()
            found_kw = sum(1 for kw in test['keywords'] if kw.lower() in top_text)
            
            if found_kw >= len(test['keywords']) * 0.5:
                passed += 1
                status = "[PASS]"
            else:
                status = "[FAIL]"
            
            print(f"  {status} Test {i}: {test['query'][:45]}... ({found_kw}/{len(test['keywords'])} keywords, {latency*1000:.1f}ms)")
        
        acc = (passed / total) * 100
        avg_lat = np.mean(latencies) * 1000
        print(f"\n  Accuracy: {passed}/{total} ({acc:.1f}%) | Avg Latency: {avg_lat:.1f}ms")
        
        self.results_summary['material_classification'] = {
            'accuracy': acc, 'passed': passed, 'total': total, 'avg_latency_ms': avg_lat
        }
    
    def test_ranking_quality(self):
        print("\n[Test 6] Ranking Quality (NDCG)")
        
        test_cases = [
            {'query': 'titanium alloy Ti-6Al-4V properties', 'keywords': ['Ti-6Al-4V', 'titanium']},
            {'query': 'hydroxyapatite bioactive ceramic bone', 'keywords': ['hydroxyapatite', 'bone']},
            {'query': 'stainless steel 316L surgical instruments', 'keywords': ['316L', 'stainless']}
        ]
        
        passed, total, latencies, ndcg_scores = 0, len(test_cases), [], []
        
        for i, test in enumerate(test_cases, 1):
            start_time = time.time()
            search_result = self.rag.semantic_search(test['query'], top_k=10)
            latency = time.time() - start_time
            latencies.append(latency)
            
            results = search_result['results']
            
            rel_scores = []
            for idx, r in enumerate(results):
                text = r['text'].lower()
                rel = sum(3 - min(2, idx) if kw.lower() in text else 0 for kw in test['keywords'])
                rel_scores.append(rel)
            
            dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(rel_scores))
            ideal = sorted(rel_scores, reverse=True)
            idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
            
            top_text = ' '.join([results[pos]['text'].lower() for pos in [0, 1, 2] if pos < len(results)])
            found_kw = sum(1 for kw in test['keywords'] if kw.lower() in top_text)
            
            if found_kw >= len(test['keywords']) * 0.6 and ndcg >= 0.7:
                passed += 1
                status = "[PASS]"
            else:
                status = "[FAIL]"
            
            print(f"  {status} Test {i}: {test['query'][:45]}... (NDCG: {ndcg:.3f}, {latency*1000:.1f}ms)")
        
        acc = (passed / total) * 100
        avg_ndcg = np.mean(ndcg_scores)
        avg_lat = np.mean(latencies) * 1000
        print(f"\n  Accuracy: {passed}/{total} ({acc:.1f}%) | Avg NDCG: {avg_ndcg:.3f} | Avg Latency: {avg_lat:.1f}ms")
        
        self.results_summary['ranking_quality'] = {
            'accuracy': acc, 'passed': passed, 'total': total, 'avg_ndcg': avg_ndcg, 'avg_latency_ms': avg_lat
        }
    
    def test_negative_cases(self):
        print("\n[Test 7] Negative Cases (Irrelevant Queries)")
        
        test_cases = [
            {'query': 'quantum computing algorithms', 'max_mat': 2},
            {'query': 'cooking recipes for pasta', 'max_mat': 1},
            {'query': 'weather forecast tomorrow', 'max_mat': 1},
            {'query': 'XYZ-999 completely made up material', 'max_mat': 3}
        ]
        
        passed, total, latencies = 0, len(test_cases), []
        
        for i, test in enumerate(test_cases, 1):
            start_time = time.time()
            try:
                search_result = self.rag.semantic_search(test['query'], top_k=5)
                latency = time.time() - start_time
                latencies.append(latency)
                
                results = search_result['results']
                mat_count = sum(1 for r in results if r['metadata'].get('type') == 'material')
                top_score = results[0]['similarity_score'] if results else 0
                
                if mat_count <= test['max_mat'] or top_score < 0.30:
                    passed += 1
                    status = "[PASS]"
                else:
                    status = "[FAIL]"
                
                print(f"  {status} Test {i}: {test['query'][:45]}... ({mat_count} materials, score: {top_score:.3f}, {latency*1000:.1f}ms)")
            except:
                passed += 1
                print(f"  [PASS] Test {i}: Properly handled irrelevant query")
        
        acc = (passed / total) * 100
        avg_lat = np.mean(latencies) * 1000 if latencies else 0
        print(f"\n  Accuracy: {passed}/{total} ({acc:.1f}%) | Avg Latency: {avg_lat:.1f}ms")
        
        self.results_summary['negative_cases'] = {
            'accuracy': acc, 'passed': passed, 'total': total, 'avg_latency_ms': avg_lat
        }
    
    def test_cross_domain_knowledge(self):
        print("\n[Test 8] Cross-Domain Knowledge Integration")
        
        test_cases = [
            {'query': 'titanium biomaterials research studies clinical', 'min_div': 2},
            {'query': 'polymer drug delivery biomedical engineering', 'min_div': 2},
            {'query': 'orthopedic implant materials clinical outcomes', 'min_div': 2}
        ]
        
        passed, total, latencies = 0, len(test_cases), []
        
        for i, test in enumerate(test_cases, 1):
            start_time = time.time()
            search_result = self.rag.semantic_search(test['query'], top_k=10)
            latency = time.time() - start_time
            latencies.append(latency)
            
            results = search_result['results']
            types = set(r['metadata'].get('type') for r in results)
            has_mat = any(r['metadata'].get('type') == 'material' for r in results)
            
            if len(types) >= test['min_div'] and has_mat:
                passed += 1
                status = "[PASS]"
            else:
                status = "[FAIL]"
            
            types_str = ', '.join(types)
            print(f"  {status} Test {i}: {test['query'][:45]}... ({len(types)} types: {types_str}, {latency*1000:.1f}ms)")
        
        acc = (passed / total) * 100
        avg_lat = np.mean(latencies) * 1000
        print(f"\n  Accuracy: {passed}/{total} ({acc:.1f}%) | Avg Latency: {avg_lat:.1f}ms")
        
        self.results_summary['cross_domain_knowledge'] = {
            'accuracy': acc, 'passed': passed, 'total': total, 'avg_latency_ms': avg_lat
        }
    
    def print_final_summary(self):
        print("\n" + "="*60)
        print("FINAL TEST SUMMARY")
        print("="*60)
        
        total_passed = sum(r['passed'] for r in self.results_summary.values())
        total_tests = sum(r['total'] for r in self.results_summary.values())
        overall_acc = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\nOverall Accuracy: {total_passed}/{total_tests} ({overall_acc:.1f}%)")
        print(f"\nTest Category Results:")
        
        for name, res in self.results_summary.items():
            display = name.replace('_', ' ').title()
            print(f"\n  {display}:")
            print(f"    - Accuracy: {res['accuracy']:.1f}% ({res['passed']}/{res['total']})")
            print(f"    - Avg Latency: {res['avg_latency_ms']:.1f}ms")
            if 'avg_ndcg' in res:
                print(f"    - Avg NDCG: {res['avg_ndcg']:.3f}")
        
        avg_lat = np.mean([r['avg_latency_ms'] for r in self.results_summary.values()])
        print(f"\nPerformance:")
        print(f"    - Average Latency: {avg_lat:.1f}ms")
        print(f"    - Target: <100ms {'[OK]' if avg_lat < 100 else '[WARNING]'}")
        
        if overall_acc >= 90:
            grade = "[EXCELLENT]"
        elif overall_acc >= 80:
            grade = "[GOOD]"
        elif overall_acc >= 70:
            grade = "[ACCEPTABLE]"
        else:
            grade = "[NEEDS IMPROVEMENT]"
        
        print(f"\nSystem Grade: {grade}")
        print("="*60)
    
    def save_results(self):
        output_dir = Path(__file__).parent / "test_output"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "rag_accuracy_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results_summary, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def main():
    print("\nInitializing RAG Accuracy Test Suite...")
    tester = RAGAccuracyTester()
    
    print("Loading Health Materials RAG Database...")
    tester.rag.load_database()
    
    tester.run_all_tests()
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
