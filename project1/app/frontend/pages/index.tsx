import Link from 'next/link'
import Layout from '../components/Layout'

const IndexPage = () => (
  <Layout title="Home | Next.js + TypeScript Example">
    <h1>Hello Next.js 👋</h1>
    <p>
      <Link href="/about">
        <a>About</a>
      </Link>
    </p>
  </Layout>
)

export async function getServerSideProps(context) {
  const res = await fetch()
  return {
    props: {}, // will be passed to the page component as props
  }
}

export default IndexPage;
