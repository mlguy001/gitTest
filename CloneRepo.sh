function createBareRepo {
	TargetRepoLocation="https://github.com/mlguy001"
	RepoName="gitTest"
	
	WorkspaceLocation="C:\Users\Nidhal\Desktop\Classes\Git testing"
	#WorkspaceName="Workspace $FolderName"
	WorkspaceName="WorkspaceGitTest"
	
	echo "$TargetRepoLocation/$RepoName/"
			
	if [ -d "$TargetRepoLocation/$RepoName/" ];
			then
			
				git clone "$TargetRepoLocation/$RepoName/" "$WorkspaceLocation/$WorkspaceName/"

	else
				echo "$RepoName doesn't exist"
	fi
}

createBareRepo

read -p ""