import pkg_resources

dist = pkg_resources.get_distribution("streamlit_authenticator")
print("Library:", dist.project_name)
print("Version:", dist.version)
print("Requires:", [str(req) for req in dist.requires()])