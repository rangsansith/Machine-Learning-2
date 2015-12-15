package hw1.decisiontree.main;

import hw1.decisiontree.datatype.CustomTree;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

public class ClassificationTree {

	private static Double CalculateEntropy(HashMap<Double, Integer> count,int total)
	{
		double prob=0.0,entropy=0.0;
		for (Double key:count.keySet()) {
			prob=count.get(key)/(double) total;
			if(prob!=0)
				entropy+=(-1)*prob*(Math.log10(prob)/Math.log10(2));
		}
		return entropy;
	}
	private static Double InformationGain(Double entropy,HashMap<Double, Integer> less,HashMap<Double, Integer> notless,int l,int nl,int total){
		return entropy-(double)l/(double)total*CalculateEntropy(less, l)-(double)nl/(double)total*CalculateEntropy(notless, nl);
	}
	private static Double[] FindThreshold(ArrayList<Double> labels,ArrayList<Double> feat){
		int total=labels.size(),l=0,nl=total;
		Map<Double, HashMap<Double,Integer>> featlabel = new HashMap<Double,HashMap<Double,Integer>>();
		int eachval=0;
		Double featkey=0.0;
		for(int i=0;i<total;i++)
		{
			HashMap<Double,Integer> tempval = featlabel.get(feat.get(i));
			featkey=labels.get(i);
			if(tempval==null)
			{
				tempval=new HashMap<Double,Integer>();
				tempval.put(featkey, 1);
			}
			else
			{
				eachval=0;
				if(tempval.containsKey(featkey))
				{
					eachval=tempval.get(featkey);
				}
				tempval.put(featkey, eachval+1);
			}
			featlabel.put(feat.get(i),tempval);
		}
		featlabel=new TreeMap<Double,HashMap<Double,Integer>>(featlabel);
		HashMap<Double, Integer> count=new HashMap<Double, Integer>();
		double temp=0.0;
		for (int i = 0; i < total; i++) {
			temp=labels.get(i);
			if(count.containsKey(temp))
			{
				count.put(temp,(count.get(temp)+1));
			}
			else
			{
				count.put(temp,1);
			}
		}
		temp=CalculateEntropy(count, total);
		double ig=0.0,curthreshold=0.0,threshold=0.0,temp1=0.0,temp2=0.0,tempig=0.0;
		HashMap<Double, Integer> less=new HashMap<Double, Integer>();
		for(Double key:count.keySet())
		{
			less.put(key, 0);
		}
		int i=0;
		//System.out.println(featlabel.size());
		for (Double key:featlabel.keySet()) {
			HashMap<Double,Integer> tempval = featlabel.get(key);
			if(i==0)
			{
				for (Double tempvalkey:tempval.keySet()) 
				{
					less.put(tempvalkey,less.get(tempvalkey)+tempval.get(tempvalkey));
					count.put(tempvalkey,(count.get(tempvalkey)-tempval.get(tempvalkey)));
					l+=tempval.get(tempvalkey);nl-=tempval.get(tempvalkey);
					//System.out.println("here  "+key+"   "+tempvalkey+"   "+less.get(tempvalkey)+"  "+count.get(tempvalkey)+"  "+tempval.get(tempvalkey));
				}
				temp1=key;
				i++;
				continue;
			}
			temp2=key;
			curthreshold=(temp1+temp2)/2.0;
			tempig=InformationGain(temp, less, count, l, nl, total);
			//System.out.println(curthreshold+"   "+l+"  "+nl+"  "+total+"  "+tempig);
			if(tempig>ig)
			{
				ig=tempig;
				threshold=curthreshold;
			}
			temp1=temp2;
			for (Double tempvalkey:tempval.keySet()) 
			{
				less.put(tempvalkey,less.get(tempvalkey)+tempval.get(tempvalkey));
				count.put(tempvalkey,(count.get(tempvalkey)-tempval.get(tempvalkey)));
				l+=tempval.get(tempvalkey);nl-=tempval.get(tempvalkey);
				//System.out.println("here  "+key+"   "+tempvalkey+"   "+less.get(tempvalkey)+"  "+count.get(tempvalkey)+"  "+tempval.get(tempvalkey));
			}

		}
		Double[] all={threshold,ig};
		return all;
	}
	private static Double[] FindBestSplit(ArrayList<Double> labels,ArrayList<ArrayList<Double>> feats){
		double ig=0.0,curfeat=0.0,tempig=0.0,thresh=0.0,tempthresh=0.0;
		Double[] result={0.0,0.0};
		for (int i = 0; i < feats.size(); i++) {
			//System.out.println(i);
			result=FindThreshold(labels,feats.get(i));
			tempig=result[1];
			tempthresh=result[0];
			//System.out.println(tempig+"   "+tempthresh);
			if(tempig>ig)
			{
				ig=tempig;
				curfeat=i;
				thresh=tempthresh;
			}
		}
		Double[] all={curfeat,thresh,ig};
		return all;
	}
	private static Double[] CreateFeatMat(ArrayList<Double> labels,ArrayList<Integer> index,ArrayList<ArrayList<Double>> feats){
		ArrayList<ArrayList<Double>> featuresnew = new ArrayList<ArrayList<Double>>();
		ArrayList<Double> labelsnew = new ArrayList<Double>();
		for(int i=0;i<index.size();i++)
		{
			labelsnew.add(i,labels.get(index.get(i)));
		}
		for (int i = 0; i < feats.size(); i++) {
			ArrayList<Double> temp = new ArrayList<Double>();
			for(int j=0;j<index.size();j++)
			{
				//System.out.println(i+" "+index.get(j)+" "+feats.size());
				temp.add(j,feats.get(i).get(index.get(j)));
			}
			featuresnew.add(i,temp);
		}
		return FindBestSplit(labelsnew, featuresnew);
	}
	private static ArrayList<ArrayList<Integer>> SplitOnThreshold(Double threshold,ArrayList<Double> feat,ArrayList<Integer> index){
		ArrayList<Integer> lindex = new ArrayList<Integer>();
		ArrayList<Integer> rindex = new ArrayList<Integer>();
		int lind=0,rind=0;
		for (int i = 0; i < index.size(); i++) {
			//System.out.println(i);
			if(feat.get(index.get(i))<threshold)
			{
				//System.out.println(index.get(i));
				lindex.add(lind,index.get(i));
				lind++;
			}
			else
			{
				rindex.add(rind,index.get(i));
				rind++;	
			}
		}
		ArrayList<ArrayList<Integer>> all = new ArrayList<ArrayList<Integer>>(); 
		all.add(0,lindex);
		all.add(1,rindex);
		return all;
	}
	private static Double MaxFeature(ArrayList<Double> labels,ArrayList<Integer> index){
		int max=0;double ans=0.0;
		HashMap<Double, Integer> maxc = new HashMap<Double, Integer>();
		for (int i = 0; i < index.size(); i++) {
			if(maxc.containsKey(labels.get(index.get(i))))
			{
				maxc.put(labels.get(index.get(i)), maxc.get(labels.get(index.get(i)))+1);
			}
			else
				maxc.put(labels.get(index.get(i)), 1);
		}
		//System.out.println("size "+maxc.size()+"   "+index.size());
		for(Double key:maxc.keySet())
		{
			//System.out.println(key+"    "+maxc.get(key));
			if(maxc.get(key)>max)
			{
				max=maxc.get(key);
				ans=key;
			}
		}
		return ans;
	}
	private static CustomTree BuildNode(ArrayList<ArrayList<Double>> features,ArrayList<Double> labels,ArrayList<Integer> index,int nodeno){
		Double[] all= CreateFeatMat(labels, index, features);
		CustomTree node = new CustomTree(nodeno,all[1],all[0].intValue(),MaxFeature(labels, index));
		//System.out.println(nodeno+"   "+node.getLabel()+" "+node.getType()+"   "+node.getThreshold());
		return node;
	}
	private static CustomTree BuildTree(ArrayList<ArrayList<Double>> features,ArrayList<Double> labels,HashMap<Integer, ArrayList<Integer>> indexlist){
		ArrayList<Integer> nodelist = new ArrayList<Integer>();
		int nodeno=1;
		nodelist.add(nodeno);
		CustomTree tree = new CustomTree();
		for (int i = 0; i < 103; i++) {
			nodeno=nodelist.get(i);
			CustomTree node = new CustomTree();
			node=BuildNode(features, labels, indexlist.get(nodeno), nodeno);
			//System.out.println(i+" a  here "+node.getNodeno());
			if(i==0)
				tree.getNode(node);
			else
				tree.placenode(nodeno, node);
			//.out.println(i+" b  here "+tree.getNodeno());
			ArrayList<ArrayList<Integer>> ans = SplitOnThreshold(node.getThreshold(), features.get(node.getFeatno()), indexlist.get(nodeno));
			//.out.println(i+" c  here ");
			//System.out.println(node.getFeatno());
			//System.out.println(i+"   "+ans.get(0).size()+"   "+ans.get(1).size());
			indexlist.put(nodeno*2, ans.get(0));
			indexlist.put(nodeno*2+1, ans.get(1));
			nodelist.add(nodeno*2);
			nodelist.add(nodeno*2+1);
		}
		return tree;
	}
	private static Double[] AccuracyPredict(ArrayList<ArrayList<Double>> featurestest,ArrayList<Double> labelstest,CustomTree tree){
		int total=labelstest.size();
		int wrong=0;
		double mse=0.0;
		for (int i = 0; i < total; i++){
			double thresh=0.0,predict=0.0;int featno=0;
			CustomTree ttemp = new CustomTree();
			ttemp=tree;
			while(true)
			{
				featno=ttemp.getFeatno();
				thresh=ttemp.getThreshold();
				//System.out.println(ttemp.getNodeno()+"   "+featno+"   "+thresh+"   "+featurestest.get(featno).get(i)+"   "+ttemp.getType()+"   "+ttemp.getLabel());
				if(ttemp.getType().contentEquals("tree"))
				{
					if(featurestest.get(featno).get(i)<thresh)
					{
						ttemp=ttemp.getLeft();
					}
					else
					{
						ttemp=ttemp.getRight();
					}
				}
				else
				{
					predict=ttemp.getLabel();
					break;
				}
			}
			if(labelstest.get(i)!=predict)
				{
				  wrong++;
				  mse+=(predict-labelstest.get(i))*(predict-labelstest.get(i));
				}
		}
		Double all[]={mse/(double) total,(100-wrong*100/(double) total),(double) wrong};
		return all;
	}
	public static void main(String[] args) {

		ArrayList<ArrayList<Double>> features = new ArrayList<ArrayList<Double>>();
		ArrayList<Double> labels = new ArrayList<Double>();
		ArrayList<Integer> index = new ArrayList<Integer>();
		HashMap<Integer, ArrayList<Integer>> indexlist = new HashMap<Integer, ArrayList<Integer>>();
		FileReader featureread;
		try {
			featureread = new FileReader("data/spambase.data");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			sCurrentLine = featurereadbr.readLine();
			feats=sCurrentLine.split(",");
			for(int i=0;i<feats.length-1;i++)
			{
				features.add(i, new ArrayList<Double>());
			}
			for(int i=0;i<feats.length-1;i++)
			{
				ArrayList<Double> temp=features.get(i);
				temp.add(ind, Double.parseDouble(feats[i]));
				features.set(i, temp);
			}
			labels.add(ind, Double.parseDouble(feats[feats.length-1]));
			index.add(ind);
			ind++;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				feats=sCurrentLine.split(",");
				for(int i=0;i<feats.length-1;i++)
				{
					ArrayList<Double> temp=features.get(i);
					temp.add(ind, Double.parseDouble(feats[i]));
					features.set(i, temp);
				}
				labels.add(ind, Double.parseDouble(feats[feats.length-1]));
				index.add(ind);
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		ArrayList<ArrayList<ArrayList<Double>>> featklist = new ArrayList<ArrayList<ArrayList<Double>>>();
		ArrayList<ArrayList<Double>> labelsklist = new ArrayList<ArrayList<Double>>();
		int kfold=400;int listno=0;
		Random n = new Random(); 
		while(labels.size()!=0){
			ArrayList<ArrayList<Double>> featurestemp = new ArrayList<ArrayList<Double>>();
			ArrayList<Double> labelstemp = new ArrayList<Double>();
			for(int i=0;i<features.size();i++)
			{
				featurestemp.add(i, new ArrayList<Double>());
			}
			for (int i = 0; i < kfold; i++) {
				if(labels.size()==0)
					break;
				int ind=n.nextInt(labels.size());
				//System.out.println(labels.size()+"  "+ind+"  ");
				for (int j = 0; j < features.size(); j++) {
					ArrayList<Double> tempt = featurestemp.get(j);
					tempt.add(i,features.get(j).get(ind));
					features.get(j).remove(ind);
					featurestemp.set(j, tempt);
				}
				labelstemp.add(i,labels.get(ind));
				labels.remove(ind);
			}
			featklist.add(listno,featurestemp);
			labelsklist.add(listno,labelstemp);
			listno++;
		}
		int nind=0;Double sum=0.0,su=0.0,msetr=0.0,msete=0.0;
		for (int i = 0; i < listno; i++) {
			nind=0;
			ArrayList<ArrayList<Double>> featurestrain = new ArrayList<ArrayList<Double>>();
			ArrayList<Double> labelstrain = new ArrayList<Double>();
			ArrayList<Integer> indextrain = new ArrayList<Integer>();
			ArrayList<ArrayList<Double>> featurestest = new ArrayList<ArrayList<Double>>();
			ArrayList<Double> labelstest = new ArrayList<Double>();
			for(int j=0;j<features.size();j++)
			{
				featurestest.add(j, new ArrayList<Double>());
				featurestrain.add(j, new ArrayList<Double>());
			}
			for (int j = 0; j < listno; j++) {
				ArrayList<ArrayList<Double>> featurestemp = featklist.get(j);
				ArrayList<Double> labelstemp = labelsklist.get(j);
				//System.out.println("before "+i+"  "+j+"  "+nind+"  "+labelsklist.get(j).size());
				if(j==i)
				{
					for (int k = 0; k < labelstemp.size(); k++) {
						labelstest.add(labelstemp.get(k));
						for (int k2 = 0; k2 < featurestemp.size(); k2++) {
							ArrayList<Double> tempt = featurestest.get(k2);
							tempt.add(featurestemp.get(k2).get(k));
							featurestest.set(k2, tempt);
						}
					}
				}
				else
				{
					for (int k = 0; k < labelstemp.size(); k++) {
						labelstrain.add(labelstemp.get(k));
						indextrain.add(nind);
						nind++;//System.out.println(k+"  "+nind);
						for (int k2 = 0; k2 < featurestemp.size(); k2++) {
							ArrayList<Double> tempt = featurestrain.get(k2);
							tempt.add(featurestemp.get(k2).get(k));
							featurestrain.set(k2, tempt);
						}
					}
				}
				//System.out.println("after "+i+"  "+j+"  "+nind+"  "+labelsklist.get(j).size());
			}
			//System.out.println(i+"  "+nind+"  "+labelsklist.get(i).size());
			indexlist.put(1, indextrain);
			System.out.println(featurestrain.size());
			CustomTree tree = BuildTree(featurestrain, labelstrain, indexlist);
			Double allte[]=AccuracyPredict(featurestest, labelstest, tree);
			Double alltr[]=AccuracyPredict(featurestrain, labelstrain, tree);
			sum+=allte[1];
			su+=alltr[1];
			msete+=allte[0];
			msetr+=alltr[0];
			//System.out.println(allte[0]+"  "+labelstest.size()+"  "+allte[2]+"    "+allte[1]);
			//System.out.println("Accuracy: "+AccuracyPredict(featurestest, labelstest, tree));
			
			System.out.println("Iteration :"+(i+1));
			System.out.println("Testing error:"+allte[0]);
			System.out.println("Testing Accuracy:"+allte[1]);
			System.out.println("Training error:"+alltr[0]);
			System.out.println("Training Accuracy:"+alltr[1]);
		}
		//System.out.println("Avg Accuracy: "+msete+"  "+(sum/(double) listno)+"  "+(su/(double) listno)+"  "+(msete/(double) listno)+"   "+(msetr/(double) listno));
		System.out.println("Average Testing error:"+(msete/(double) listno));
		System.out.println("Average Testing Accuracy:"+(sum/(double) listno));
		System.out.println("Average Training error:"+(msetr/(double) listno));
		System.out.println("Average Training Accuracy:"+(su/(double) listno));
	}

}
