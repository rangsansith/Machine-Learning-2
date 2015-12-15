package hw1.decisiontree.main;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

import hw1.decisiontree.datatype.CustomTree;

class output{
	ArrayList<Double[]> confusion;
	double accuracy;
	double error;
	output(){
		confusion = new ArrayList<Double[]>();
		accuracy=0.0;
		error=0.0;
	}
}
public class DecisionTreeSpam {

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
	private static output AccuracyPredict(ArrayList<ArrayList<Double>> featurestest,ArrayList<Double> labelstest,CustomTree tree){
		int total=labelstest.size();
		output tmp = new output();
		int wrong=0;
		double mse=0.0;
		Double[] cc={0.0,0.0,0.0,0.0};
		for (int i = 0; i < total; i++){
			double thresh=0.0,predict=0.0;int featno=0;
			CustomTree ttemp = new CustomTree();
			ttemp=tree;
			while(true)
			{
				featno=ttemp.getFeatno();
				thresh=ttemp.getThreshold();
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
			if(labelstest.get(i)==1.0)
			{
				if(predict==1.0)
					cc[0]++;
				else
				{
					cc[3]++;
					wrong++;
					mse+=(predict-labelstest.get(i))*(predict-labelstest.get(i));
				}
			}
			else
			{
				if(predict==0.0)
					cc[2]++;
				else
				{
					cc[1]++;
					wrong++;
					mse+=(predict-labelstest.get(i))*(predict-labelstest.get(i));
				}
			}
			Double newt[]=cc.clone();
			tmp.confusion.add(i, newt);
		}
		tmp.error=mse/(double) total;
		tmp.accuracy=(((double)(total-wrong))/((double) total))*100;
		return tmp;
	}
	
	public static void main(String[] args) {
		ArrayList<ArrayList<Double>> featurestrain = new ArrayList<ArrayList<Double>>();
		ArrayList<Double> labelstrain = new ArrayList<Double>();
		ArrayList<Integer> indextrain = new ArrayList<Integer>();
		ArrayList<ArrayList<Double>> featurestest = new ArrayList<ArrayList<Double>>();
		ArrayList<Double> labelstest = new ArrayList<Double>();
		HashMap<Integer, ArrayList<Integer>> indexlist = new HashMap<Integer, ArrayList<Integer>>();
		FileReader featureread;
		try {
			featureread = new FileReader("data/spam_train");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			sCurrentLine = featurereadbr.readLine();
			feats=sCurrentLine.split(" ");
			for(int i=0;i<feats.length-1;i++)
			{
				featurestrain.add(i, new ArrayList<Double>());
			}
			for(int i=0;i<feats.length-1;i++)
			{
				ArrayList<Double> temp=featurestrain.get(i);
				temp.add(ind, Double.parseDouble(feats[i]));
				featurestrain.set(i, temp);
			}
			labelstrain.add(ind, Double.parseDouble(feats[feats.length-1]));
			indextrain.add(ind);
			ind++;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length-1;i++)
				{
					ArrayList<Double> temp=featurestrain.get(i);
					temp.add(ind, Double.parseDouble(feats[i]));
					featurestrain.set(i, temp);
				}
				labelstrain.add(ind, Double.parseDouble(feats[feats.length-1]));
				indextrain.add(ind);
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		try {
			featureread = new FileReader("data/spam_test");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			sCurrentLine = featurereadbr.readLine();
			feats=sCurrentLine.split(" ");
			for(int i=0;i<feats.length-1;i++)
			{
				featurestest.add(i, new ArrayList<Double>());
			}
			for(int i=0;i<feats.length-1;i++)
			{
				ArrayList<Double> temp=featurestest.get(i);
				temp.add(ind, Double.parseDouble(feats[i]));
				featurestest.set(i, temp);
			}
			labelstest.add(ind, Double.parseDouble(feats[feats.length-1]));
			ind++;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length-1;i++)
				{
					ArrayList<Double> temp=featurestest.get(i);
					temp.add(ind, Double.parseDouble(feats[i]));
					featurestest.set(i, temp);
				}
				labelstest.add(ind, Double.parseDouble(feats[feats.length-1]));
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		indexlist.put(1, indextrain);
		CustomTree tree = BuildTree(featurestrain, labelstrain, indexlist);
		output test = AccuracyPredict(featurestest, labelstest, tree);
		output train = AccuracyPredict(featurestrain, labelstrain, tree);
		System.out.println("Testing error:"+test.error);
		System.out.println("Testing Accuracy:"+test.accuracy);
		System.out.println("Training error:"+train.error);
		System.out.println("Training Accuracy:"+train.accuracy);
		for (int i = 0; i < test.confusion.size(); i++) {
			//System.out.println(i+"TP:"+test.confusion.get(i)[0]+"FP:"+test.confusion.get(i)[1]+"TN:"+test.confusion.get(i)[2]+"FN:"+test.confusion.get(i)[3]);
		}
		try {
			FileWriter ccdt = new FileWriter("result/spam_ccdt");
			BufferedWriter ccdtbw = new BufferedWriter(ccdt);
		      for (int i = 0; i < test.confusion.size(); i++) {
				ccdtbw.write(i+" "+test.confusion.get(i)[0]+" "+test.confusion.get(i)[1]+" "+test.confusion.get(i)[2]+" "+test.confusion.get(i)[3]+"\n");
			}
			ccdtbw.close();
			ccdt.close();
		}catch (Exception e){
			System.out.println(e.getMessage());	
		}
	}

}
