package hw4.boosting.gradient;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import hw1.decisiontree.datatype.CustomTree;

public class GradientBoost {

	static ArrayList<ArrayList<Double>> featurestrain = new ArrayList<ArrayList<Double>>();
	static ArrayList<ArrayList<Double>> labelstrain = new ArrayList<ArrayList<Double>>();
	static ArrayList<Integer> indextrain = new ArrayList<Integer>();
	static ArrayList<ArrayList<Double>> featurestest = new ArrayList<ArrayList<Double>>();
	static ArrayList<Double> labelstest = new ArrayList<Double>();
	static ArrayList<CustomTree> treelist = new ArrayList<CustomTree>();
	static int runs=10;
	private static Double CalculateMSE(ArrayList<Double> labels)
	{
		double sum=0.0;
		for (int i = 0; i < labels.size(); i++) {
			sum+=labels.get(i);
		}
		double avg=sum/(double) labels.size();
		double error=0.0;
		for (int i = 0; i < labels.size(); i++) {
			error+=(labels.get(i)-avg)*(labels.get(i)-avg);
		}
		return error;
	}
	private static Double CalculateMSEreduction(ArrayList<Double> labels,ArrayList<Double> feats,double threshold)
	{
		ArrayList<Double> less = new ArrayList<Double>();
		ArrayList<Double> nless = new ArrayList<Double>();
		int total=feats.size();
		for (int i = 0; i < total; i++) {
			if(feats.get(i)<threshold)
			{
				less.add(labels.get(i));
			}
			else
			{
				nless.add(labels.get(i));
			}
		}
		return CalculateMSE(less)+CalculateMSE(nless);
	}
	private static Double[] SelectThreshold(ArrayList<Double> labels,ArrayList<Double> feats)
	{
		ArrayList<Double> temp = new ArrayList<Double>(feats);
		Collections.sort(temp);
		double temp1=temp.get(0),temp2=0.0,thresh=-1.0,tthresh=0.0,bmre=CalculateMSE(labels),trme=0.0;
		for (int i = 1; i < temp.size(); i++) {
			temp2=temp.get(i);
			thresh=(temp1+temp2)/2.0;
			trme=CalculateMSEreduction(labels, feats, thresh);
			if(trme<bmre)
			{
				bmre=trme;
				tthresh=thresh;
			}
			temp1=temp2;
		}
		Double all[]={tthresh,bmre};
		return all;
	}
	private static Double[] FindBestSplit(ArrayList<Double> labels,ArrayList<ArrayList<Double>> feats){
		//System.out.println("bsp"+labels.size());
		double re=CalculateMSE(labels),curfeat=-1.0,tempre=0.0,thresh=-1.0,tempthresh=0.0;
		Double[] result={0.0,0.0};
		for (int i = 0; i < feats.size(); i++) {
			result=SelectThreshold(labels,feats.get(i));
			tempre=result[1];
			tempthresh=result[0];
			if(tempre<re)
			{
				re=tempre;
				curfeat=i;
				thresh=tempthresh;
			}
		}
		Double[] all={curfeat,thresh,re};
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
		//System.out.println(CalculateMSE(labelsnew)+"inga gokka");
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
		double ans=0.0;
		for (int i = 0; i < index.size(); i++) {
			ans+=labels.get(index.get(i));
		}
		return ans/(double) index.size();
	}
	private static CustomTree BuildNode(ArrayList<ArrayList<Double>> features,ArrayList<Double> labels,ArrayList<Integer> index,int nodeno){
		CustomTree node = new CustomTree();
		if(index.size()==1)
		{
			node = new CustomTree(nodeno,-1,-1,MaxFeature(labels, index));
		}
		else
		{
			Double[] all= CreateFeatMat(labels, index, features);
			//System.out.println("nodeno "+nodeno+"   "+all[0]+"  "+all[1]+"  "+all[2]);
			/*if(index.size()<5)
			{
				for (int i = 0; i < index.size(); i++) {
					for (int j = 0; j < features.size(); j++) {
						System.out.print(features.get(j).get(index.get(i))+"  ");
					}
					System.out.println(labels.get(index.get(i)));
				}
			}*/
			node = new CustomTree(nodeno,all[1],all[0].intValue(),MaxFeature(labels, index));
			//System.out.println(nodeno+"   "+node.getLabel()+" "+node.getType()+"   "+node.getThreshold());
		}
		return node;
	}
	private static CustomTree BuildTree(ArrayList<ArrayList<Double>> features,ArrayList<Double> labels,HashMap<Integer, ArrayList<Integer>> indexlist){
		ArrayList<Integer> nodelist = new ArrayList<Integer>();
		int nodeno=1;
		nodelist.add(nodeno);
		CustomTree tree = new CustomTree();
		for (int i = 0; i < 7; i++) {
			nodeno=nodelist.get(i);
			CustomTree node = new CustomTree();
			//System.out.println("HOI"+nodeno+"   "+indexlist.get(nodeno).size());
			node=BuildNode(features, labels, indexlist.get(nodeno), nodeno);
			//System.out.println(i+" a  here "+node.getNodeno());
			if(i==0)
				tree.getNode(node);
			else
				tree.placenode(nodeno, node);
			//.out.println(i+" b  here "+tree.getNodeno());
			//System.out.println(node.getFeatno());
			if(node.getFeatno()!=-1)
			{

				ArrayList<ArrayList<Integer>> ans = SplitOnThreshold(node.getThreshold(), features.get(node.getFeatno()), indexlist.get(nodeno));
				//.out.println(i+" c  here ");
				//System.out.println(i+"   "+ans.get(0).size()+"   "+ans.get(1).size());
				//System.out.println(nodeno+"   "+ans.get(0).size()+"   "+ans.get(1).size()+"  "+node.getFeatno());
				if(nodeno==3){

				}
				indexlist.put(nodeno*2, ans.get(0));
				indexlist.put(nodeno*2+1, ans.get(1));
				nodelist.add(nodeno*2);
				nodelist.add(nodeno*2+1);
			}
		}
		return tree;
	}
	private static Double[] AccuracyPredict(ArrayList<ArrayList<Double>> featurestest,ArrayList<Double> labelstest,CustomTree tree,int runo,boolean training){
		int total=labelstest.size();
		int wrong=0;
		double mse=0.0;
		ArrayList<Double> newtrain=new ArrayList<Double>();
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
			if(training)
			newtrain.add(i, labelstest.get(i)-predict);
			if(labelstest.get(i)!=predict)
			{
				wrong++;
				mse+=(predict-labelstest.get(i))*(predict-labelstest.get(i));
			}
		}
		if(training)
		labelstrain.add(runo, newtrain);
		Double all[]={mse/(double) total,(100-wrong*100/(double) total)};
		return all;
	}
	private static Double[] AccuracyGradPredict(ArrayList<ArrayList<Double>> featurestest,ArrayList<Double> labelstest){
		int total=labelstest.size();
		int wrong=0;
		double mse=0.0;
		for (int i = 0; i < total; i++){
			double totpredict=0.0;
			for (int j = 0; j < runs; j++) {
				CustomTree ttemp = new CustomTree();
				ttemp=treelist.get(j);
				double thresh=0.0,predict=0.0;int featno=0;
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
				totpredict+=predict;
			}
			if(labelstest.get(i)!=totpredict)
			{
				wrong++;
				mse+=(totpredict-labelstest.get(i))*(totpredict-labelstest.get(i));
			}
		}
		Double all[]={mse/(double) total,(100-wrong*100/(double) total)};
		return all;
	}
	private static void readData(){
		ArrayList<Double> trainlabel = new ArrayList<Double>();
		FileReader featureread;
		try {
			featureread = new FileReader("data/housing_train.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			sCurrentLine = featurereadbr.readLine();
			sCurrentLine = sCurrentLine.trim().replaceAll(" +", " ");
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
			trainlabel.add(ind, Double.parseDouble(feats[feats.length-1]));
			indextrain.add(ind);
			ind++;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				sCurrentLine = sCurrentLine.trim().replaceAll(" +", " ");
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length-1;i++)
				{
					ArrayList<Double> temp=featurestrain.get(i);
					temp.add(ind, Double.parseDouble(feats[i]));
					featurestrain.set(i, temp);
				}
				trainlabel.add(ind, Double.parseDouble(feats[feats.length-1]));
				indextrain.add(ind);
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		labelstrain.add(0,trainlabel);
		try {
			featureread = new FileReader("data/housing_test.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			sCurrentLine = featurereadbr.readLine();
			sCurrentLine = sCurrentLine.trim().replaceAll(" +", " ");
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
				sCurrentLine = sCurrentLine.trim().replaceAll(" +", " ");
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
	}
	public static void main(String[] args) {
		readData();
		for (int i = 0; i < runs; i++) {
			HashMap<Integer, ArrayList<Integer>> indexlist = new HashMap<Integer, ArrayList<Integer>>();
			indexlist.put(1, indextrain);
			treelist.add(i,BuildTree(featurestrain, labelstrain.get(i), indexlist));
			Double res[]=AccuracyPredict(featurestrain, labelstrain.get(i), treelist.get(i),i+1,true);
			System.out.println("Training:"+res[0]);
		}
		System.out.println("Training completed");
		Double res[]=AccuracyGradPredict(featurestest, labelstest);
		System.out.println("Testing:"+res[0]);
		res=AccuracyGradPredict(featurestrain, labelstrain.get(0));
		System.out.println("Training:"+res[0]);
	}

}
